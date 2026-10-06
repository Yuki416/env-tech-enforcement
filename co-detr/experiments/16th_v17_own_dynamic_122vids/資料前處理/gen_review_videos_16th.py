#!/usr/bin/env python3
"""產出第 16 次訓練（v17 動態子集，HEAD=15）的人工 review 影片，每個 clip 一部 mp4、疊 GT 框。

輸出結構（依 v17 的實際變動分組，讓使用者優先看「這次真正動到的」）：

  連續片段影片/
  ├── train/
  │   ├── 重標_7部/        ← 本次重標的 7 部（重點看：重標 + 收緊 HEAD 後的動態片段對不對）
  │   └── 其餘_92部/       ← 其餘 train（沿用 v16，但 HEAD 30→15，範圍也收緊了）
  ├── valid/
  │   ├── 新增_1部/        ← 本次新增到 valid 的那部 garbage clip（重點看）
  │   └── 其餘_14部/       ← 其餘 valid
  ├── test_完整8部_未過濾/  ← test 不過濾 → 放完整 clip
  └── 需重點觀察_無動態種子_N部_完整clip/ ← 無自動種子的無後綴 clip（本次 build 為 0，防呆保留）

⚠️ 影片性質：train/valid = **擷取後**的動態子集（模型實際會看到的幀，HEAD=15/TAIL=5）；
   test = **完整** clip（不過濾）；需重點觀察 = **完整** clip + 位移診斷。

host 直接跑（host 有 cv2）：PYTHONUTF8=1 python3 gen_review_videos_16th.py [LIMIT]
"""
import argparse
import cv2
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path("/path/to/data/coco/chiayi_v17_garbage_litter_refined_122vids")   # 完整 v17
DYN = Path("/path/to/data/coco/chiayi_v17_own_dynamic_litter")               # 動態子集(HEAD=15)
OUT = Path("/path/to/data/16th_expt_v17_own_dynamic_122vids_36e/連續片段影片")

FPS = 8
SEED_THRESH = 20      # 與 build_v17_dynamic_litter.py 一致
GAP_CAP = 15
MATCH_GATE_FRAC = 0.20
TARGET_CATS = [2, 5]

# R2 標註修正輪（2026-07-25）就地重標的 4 部 —— 這輪的重點。
# ⚠️ 必須用 (時間戳, 片長) 精確比對：同一時間戳可能有多個不同片長的 clip
#    （例 20240902175511 同時有 00m34s 與 01m08s，只有 01m08s 是 R2）。見 README「標註修正輪 R2」。
R2_TRAIN_KEYS = {("20240902175511", "01m08s"),
                 ("20240624173521", "06m19s"),
                 ("20240809160023", "00m25s")}                 # 3 部 train
R2_VALID_KEYS = {("20240908065812", "02m13s")}                 # 1 部 valid（person −111）
NEW_VALID_KEYS = {("20240828064620", "00m19s")}                # 首建新增的那部 garbage

_KEY_RE = re.compile(r"_(\d{14})_((?:\d{2}h)?\d{2}m\d{2}s)")


def clip_key(clip):
    m = _KEY_RE.search(clip)
    return (m.group(1), m.group(2)) if m else None

CAT = {
    1: ("car", (160, 160, 160), 2),
    2: ("litter", (0, 255, 0), 3),
    3: ("motorcycle", (255, 128, 0), 2),
    4: ("person", (0, 165, 255), 2),
    5: ("garbage", (0, 0, 255), 3),
}
frame_re = re.compile(r"^(.*)_frame_(\d+)\.jpg$", re.IGNORECASE)
LIMIT = None   # 由 --limit 設定


def sanitize(s):
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in s)


def is_new_clip(clip):
    return not clip.endswith("_RoboFlow")


def load(ds, split):
    j = json.load(open(ds / "annotations" / f"{split}.json", encoding="utf-8"))
    imgs = {im["id"]: im for im in j["images"]}
    anns = defaultdict(list)
    for a in j["annotations"]:
        anns[a["image_id"]].append(a)
    clips = defaultdict(list)
    for im in j["images"]:
        m = frame_re.match(im["file_name"])
        if m:
            clips[m.group(1)].append((int(m.group(2)), im["id"]))
    for v in clips.values():
        v.sort()
    return imgs, anns, clips


def frame_max_disp(imgs, anns, frames):
    """{frame_idx: 最大位移 px/幀}（litter+garbage 取最大），供無種子 clip 診斷。"""
    by_img = {cat: defaultdict(list) for cat in TARGET_CATS}
    for _, iid in frames:
        for a in anns[iid]:
            if a["category_id"] in TARGET_CATS:
                by_img[a["category_id"]][iid].append(a["bbox"])
    disp = defaultdict(float)
    for cat in TARGET_CATS:
        cf = sorted((f, i) for f, i in frames if i in by_img[cat])
        for k, (f_idx, iid) in enumerate(cf):
            im = imgs[iid]
            gate = MATCH_GATE_FRAC * (im["width"] ** 2 + im["height"] ** 2) ** 0.5
            nxt = next((cf[j2] for j2 in range(k + 1, len(cf))
                        if 0 < cf[j2][0] - f_idx <= GAP_CAP), None)
            if nxt is None:
                continue
            d = nxt[0] - f_idx
            dst = [(b[0] + b[2] / 2, b[1] + b[3] / 2) for b in by_img[cat][nxt[1]]]
            for b in by_img[cat][iid]:
                cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
                best = min(((cx - x) ** 2 + (cy - y) ** 2) ** 0.5 for x, y in dst)
                if best <= gate:
                    disp[f_idx] = max(disp[f_idx], best / d)
    return disp


def make_video(clip, frames, imgs, anns, img_dir, outdir, show_disp=False):
    first = cv2.imread(str(img_dir / imgs[frames[0][1]]["file_name"]))
    if first is None:
        print(f"  ⚠️ 讀不到首幀，跳過: {clip}", flush=True)
        return False
    h, w = first.shape[:2]
    outdir.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(outdir / f"{sanitize(clip)[:90]}.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    disp = frame_max_disp(imgs, anns, frames) if show_disp else {}
    peak = 0.0
    for fidx, iid in frames:
        img = cv2.imread(str(img_dir / imgs[iid]["file_name"]))
        if img is None:
            continue
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        n_lit = n_gar = 0
        for a in anns[iid]:
            nm, color, th = CAT[a["category_id"]]
            x, y, bw, bh = [int(v) for v in a["bbox"]]
            cv2.rectangle(img, (x, y), (x + bw, y + bh), color, th)
            n_lit += a["category_id"] == 2
            n_gar += a["category_id"] == 5
        bar_h = 62 if show_disp else 30
        cv2.rectangle(img, (0, 0), (w, bar_h), (0, 0, 0), -1)
        cv2.putText(img, f"f{fidx}  litter={n_lit}  garbage={n_gar}  (green=litter red=garbage)",
                    (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        if show_disp:
            d = disp.get(fidx, 0.0)
            peak = max(peak, d)
            col = (0, 255, 0) if d >= SEED_THRESH else (0, 165, 255)
            cv2.putText(img, f"max disp={d:5.1f} px/f   (seed threshold={SEED_THRESH})",
                        (6, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        vw.write(img)
    vw.release()
    extra = f"  peak_disp={peak:.1f}px/f" if show_disp else ""
    print(f"  {sanitize(clip)[:70]}.mp4  frames={len(frames)}{extra}", flush=True)
    return True


def main():
    made = [0]

    def spend():
        made[0] += 1
        return LIMIT is not None and made[0] >= LIMIT

    # ---- 1. train：擷取後動態片段，分「R2重標_3部」/「其餘」----
    imgs, anns, clips = load(DYN, "train")
    present = {clip_key(c) for c in clips}
    missing = R2_TRAIN_KEYS - present
    if missing:
        print(f"⚠️ 有 R2 重標 train clip 不在動態子集（可能掉種子）：{sorted(missing)}", flush=True)
    reannot = {c: f for c, f in clips.items() if clip_key(c) in R2_TRAIN_KEYS}
    rest = {c: f for c, f in clips.items() if clip_key(c) not in R2_TRAIN_KEYS}
    print(f"\n=== train（擷取後動態片段）：R2重標 {len(reannot)} 部 / 其餘 {len(rest)} 部 ===")
    for label, group in [(f"R2重標_{len(reannot)}部", reannot),
                         (f"其餘_{len(rest)}部", rest)]:
        print(f"\n--- train/{label} ---")
        for clip, frames in sorted(group.items()):
            make_video(clip, frames, imgs, anns, DYN / "train", OUT / "train" / label)
            if spend():
                return

    # ---- 2. valid：擷取後動態片段，分「R2重標_1部」/「新增_1部」/「其餘」----
    imgs, anns, clips = load(DYN, "valid")
    r2v = {c: f for c, f in clips.items() if clip_key(c) in R2_VALID_KEYS}
    newv = {c: f for c, f in clips.items() if clip_key(c) in NEW_VALID_KEYS}
    rest_v = {c: f for c, f in clips.items()
              if clip_key(c) not in R2_VALID_KEYS and clip_key(c) not in NEW_VALID_KEYS}
    if not r2v:
        print(f"\n⚠️ R2 valid clip（02m13s）不在動態子集", flush=True)
    if not newv:
        print(f"\n⚠️ 新增 valid clip（00m19s）不在動態子集", flush=True)
    print(f"\n=== valid（擷取後動態片段）：R2重標 {len(r2v)} / 新增 {len(newv)} / 其餘 {len(rest_v)} ===")
    for label, group in [(f"R2重標_{len(r2v)}部", r2v),
                         (f"新增_{len(newv)}部", newv),
                         (f"其餘_{len(rest_v)}部", rest_v)]:
        if not group:
            continue
        print(f"\n--- valid/{label} ---")
        for clip, frames in sorted(group.items()):
            make_video(clip, frames, imgs, anns, DYN / "valid", OUT / "valid" / label)
            if spend():
                return

    # ---- 3. test：完整 clip（不過濾）----
    imgs, anns, clips = load(DYN, "test")
    print(f"\n=== test（完整，未過濾）：{len(clips)} 部 ===")
    for clip, frames in sorted(clips.items()):
        make_video(clip, frames, imgs, anns, DYN / "test", OUT / f"test_完整{len(clips)}部_未過濾")
        if spend():
            return

    # ---- 4. 無動態種子的無後綴 clip：完整 clip + 位移診斷（本次應為 0，防呆保留）----
    s_imgs, s_anns, s_clips = load(SRC, "train")
    _, _, d_clips = load(DYN, "train")
    no_seed = sorted(c for c in s_clips if is_new_clip(c) and c not in d_clips)
    if no_seed:
        print(f"\n=== ⚠️ 需重點觀察（無動態種子，完整 clip + 位移診斷）：{len(no_seed)} 部 ===")
        for clip in no_seed:
            make_video(clip, s_clips[clip], s_imgs, s_anns, SRC / "train",
                       OUT / f"需重點觀察_無動態種子_{len(no_seed)}部_完整clip", show_disp=True)
            if spend():
                return
    else:
        print("\n（無後綴 clip 全部有種子，無『需重點觀察』影片）")
    print("\nDONE")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dyn", type=str, default=str(DYN), help="動態子集資料集目錄")
    ap.add_argument("--out", type=str, default=str(OUT), help="影片輸出目錄")
    ap.add_argument("--limit", type=int, default=None, help="只產前 N 部（smoke test）")
    args = ap.parse_args()
    DYN = Path(args.dyn)
    OUT = Path(args.out)
    LIMIT = args.limit
    print(f"DYN={DYN}\nOUT={OUT}", flush=True)
    main()
