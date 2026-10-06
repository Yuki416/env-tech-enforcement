#!/usr/bin/env python3
"""產出第 15 次訓練的人工 review 影片（每個 clip 一部 mp4，疊 GT 框）。

輸出結構（依使用者要求，train 分「新增」與「v11 原有」）：

  連續片段影片/
  ├── train/
  │   ├── 新增_NN部/                    ← 這次新標的，擷取後的動態片段
  │   └── v11原有_NN部/                 ← 第 13 次已看過，內容完全相同，可略過
  ├── valid_NN部/                       ← 擷取後的動態片段（與第 13 次完全相同）
  ├── test_完整NN部_未過濾/              ← test 不過濾，放完整 clip 供了解 test 樣貌
  └── 需重點觀察_無動態種子_NN部_完整clip/ ← 自動門檻抓不到種子的新 clip，放【完整 clip】

⚠️ 三種影片性質不同：
  - train/valid = **擷取後**的動態子集（模型實際會看到的幀）
  - test = **完整** clip（test 不過濾）
  - 需重點觀察 = **完整** clip（沒有擷取結果可看；額外疊「該幀最大位移 px/幀」
    與門檻 20 的對比，讓使用者判斷為何沒觸發、是否要補進 MANUAL_DYNAMIC_V16_NEW）

host 直接跑（host 有 cv2 4.13）：PYTHONUTF8=1 python3 gen_review_videos_15th.py [LIMIT]
"""
import cv2
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

SRC = Path("/path/to/data/coco/chiayi_v16_garbage_litter_121vids")        # 完整 v16
DYN = Path("/path/to/data/coco/chiayi_v16_own_dynamic_litter")            # 動態子集
OUT = Path("/path/to/data/15th_expt_v16_own_dynamic_121vids_36e/連續片段影片")

FPS = 8
SEED_THRESH = 20      # 與 build_v16_dynamic_litter.py 一致
GAP_CAP = 15
MATCH_GATE_FRAC = 0.20
TARGET_CATS = [2, 5]

CAT = {
    1: ("car", (160, 160, 160), 2),
    2: ("litter", (0, 255, 0), 3),
    3: ("motorcycle", (255, 128, 0), 2),
    4: ("person", (0, 165, 255), 2),
    5: ("garbage", (0, 0, 255), 3),
}
frame_re = re.compile(r"^(.*)_frame_(\d+)\.jpg$", re.IGNORECASE)
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else None


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
    """回傳 {frame_idx: 最大位移 px/幀}（litter+garbage 取最大），供無種子 clip 診斷。"""
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
    budget = LIMIT
    made = 0

    def spend(n=1):
        nonlocal made
        made += n
        return budget is not None and made >= budget

    # ---- 1. train：擷取後的動態片段，分「新增」/「v11原有」----
    imgs, anns, clips = load(DYN, "train")
    new_clips = {c: f for c, f in clips.items() if is_new_clip(c)}
    old_clips = {c: f for c, f in clips.items() if not is_new_clip(c)}
    print(f"\n=== train（擷取後動態片段）：新增 {len(new_clips)} 部 / v11原有 {len(old_clips)} 部 ===")
    for label, group in [(f"新增_{len(new_clips)}部", new_clips),
                         (f"v11原有_{len(old_clips)}部", old_clips)]:
        print(f"\n--- train/{label} ---")
        for clip, frames in sorted(group.items()):
            make_video(clip, frames, imgs, anns, DYN / "train", OUT / "train" / label)
            if spend():
                return

    # ---- 2. valid：擷取後的動態片段 ----
    imgs, anns, clips = load(DYN, "valid")
    print(f"\n=== valid（擷取後動態片段）：{len(clips)} 部 ===")
    for clip, frames in sorted(clips.items()):
        make_video(clip, frames, imgs, anns, DYN / "valid", OUT / f"valid_{len(clips)}部")
        if spend():
            return

    # ---- 3. test：完整 clip（test 不過濾）----
    imgs, anns, clips = load(DYN, "test")
    print(f"\n=== test（完整，未過濾）：{len(clips)} 部 ===")
    for clip, frames in sorted(clips.items()):
        make_video(clip, frames, imgs, anns, DYN / "test", OUT / f"test_完整{len(clips)}部_未過濾")
        if spend():
            return

    # ---- 4. 無動態種子的新 clip：完整 clip + 位移診斷 ----
    #      推導方式：v16 train 裡「新增且沒進動態子集」的 clip（不硬編清單）
    s_imgs, s_anns, s_clips = load(SRC, "train")
    _, _, d_clips = load(DYN, "train")
    no_seed = sorted(c for c in s_clips if is_new_clip(c) and c not in d_clips)
    print(f"\n=== ⚠️ 需重點觀察（無動態種子，完整 clip + 位移診斷）：{len(no_seed)} 部 ===")
    for clip in no_seed:
        make_video(clip, s_clips[clip], s_imgs, s_anns, SRC / "train",
                   OUT / f"需重點觀察_無動態種子_{len(no_seed)}部_完整clip", show_disp=True)
        if spend():
            return
    print("\nDONE")


if __name__ == "__main__":
    main()
