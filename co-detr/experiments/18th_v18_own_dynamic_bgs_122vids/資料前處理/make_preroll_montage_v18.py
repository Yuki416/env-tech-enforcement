#!/usr/bin/env python3
"""第18次 v18：pre-roll 過程可視化 montage。每部 clip 一支 mp4，一致三格版面：
  欄1 背景（pre-roll=逐幀累積 getBackgroundImage；dynamic=凍結背景）
  欄2 輸入（pre-roll=餵進 KNN 的事件前 .avi 幀；dynamic=RGB+GT 動態幀）
  欄3 changemap（pre-roll=待凍結；dynamic=該幀前景，凍結 KNN 現算，前景上綠）
→ 呈現「暖機看到什麼 → 背景凍成什麼 → 怎麼生出每幀 changemap」。導讀見 preroll_montage_導讀.md。
KNN 參數與 gen_changemaps_v18.py 完全一致（500/500/shadow, rm5, half-res）。

用法：
  單片：python make_preroll_montage_v18.py --split test --clip 20240605180713_00m20s
  全部：python make_preroll_montage_v18.py --all         （三 split 全 122 部；已存跳過）
"""
import json, re, argparse, collections
from pathlib import Path
import cv2, numpy as np

HERE  = Path(__file__).resolve().parent
V18   = Path("/path/to/data/coco/chiayi_v18_own_dynamic_bgs_122vids")
ALIGN = json.load(open(HERE / "alignment_v18.json"))
S = 0.5
HIST, DIST, RM = 500, 500.0, 5
LITTER, GARBAGE = 2, 5
TAIL_WARMUP = {"train/20240713054438_00m00s": dict(start=15, end=201)}
PW, PH, HDR = 640, 480, 74
FPS = 6
WARM_SHOW = 30
_DKERN = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def clipkey_of(name):
    """檔名 → 'ts_MMmSSs'（唯一辨識一部 clip；含小時制）。"""
    g = re.search(r"(\d{14})_(?:(\d{2})h)?(\d{2})m(\d{2})s", name)
    h = int(g.group(2)) if g.group(2) else 0
    offs = h * 3600 + int(g.group(3)) * 60 + int(g.group(4))
    return f"{g.group(1)}_{offs//60:02d}m{offs%60:02d}s"

def rm5(mask):
    fg = (mask == 255).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    out = np.zeros_like(fg)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= RM:
            out[lab == i] = 1
    return (out * 255).astype(np.uint8)

def panel(img, w=PW, h=PH):
    if img is None:
        return np.zeros((h, w, 3), np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.resize(img, (w, h))

def cmap_panel(cm, w=PW, h=PH):
    vis = cv2.dilate(cm, _DKERN)
    small = cv2.resize(vis, (w, h), interpolation=cv2.INTER_AREA)
    out = np.zeros((h, w, 3), np.uint8)
    out[small > 40] = (60, 255, 60)
    return out

def label(p, text, color=(0, 255, 255)):
    cv2.putText(p, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(p, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1, cv2.LINE_AA)
    return p

def compose(bg, mid, right, header):
    canvas = np.zeros((PH + HDR, PW * 3, 3), np.uint8)
    cv2.putText(canvas, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, header, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA)
    for i, p in enumerate((bg, mid, right)):
        canvas[HDR:HDR + PH, i * PW:(i + 1) * PW] = p
    return canvas

def load_split(split):
    """回傳 (gt_by_fn, names_by_clipkey)。每 split 只載一次。"""
    d = json.load(open(V18 / f"annotations/{split}.json"))
    id2fn = {im["id"]: im["file_name"] for im in d["images"]}
    gt = collections.defaultdict(list)
    for an in d["annotations"]:
        if an["category_id"] in (LITTER, GARBAGE):
            gt[id2fn[an["image_id"]]].append(an["bbox"])
    names_by = collections.defaultdict(list)
    for im in d["images"]:
        names_by[clipkey_of(im["file_name"])].append(im["file_name"])
    for k in names_by:
        names_by[k].sort(key=lambda n: int(re.search(r"_frame_(\d+)", n).group(1)))
    return gt, names_by

def make_one(key, gt, names, outbase, overwrite=False):
    info = ALIGN[key]
    split = info["split"]
    ck = key.split("/", 1)[1]
    outdir = Path(outbase) / split; outdir.mkdir(parents=True, exist_ok=True)
    outp = outdir / f"{ck}.mp4"
    if outp.exists() and not overwrite:
        return "skip", outp

    frs = [int(re.search(r"_frame_(\d+)", n).group(1)) for n in names]
    is_tail = key in TAIL_WARMUP
    if is_tail:
        wr = range(TAIL_WARMUP[key]["start"], TAIL_WARMUP[key]["end"]); warm_kind = "TAIL-warmup (method C)"
    else:
        wr = range(info["warm_start"], info["first_abs"]); warm_kind = "pre-roll"
    clabel = f"{split}/{ck}  win {min(frs)}-{max(frs)}/{len(names)}"

    writer = cv2.VideoWriter(str(outp), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (PW * 3, PH + HDR))

    # PHASE 1 pre-roll
    knn = cv2.createBackgroundSubtractorKNN(history=HIST, dist2Threshold=DIST, detectShadows=True)
    cap = cv2.VideoCapture(info["avi"])
    warm_idx = list(wr)
    show_every = max(1, len(warm_idx) // WARM_SHOW)
    if warm_idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, warm_idx[0])
    n_warm = 0
    waiting = label(np.full((PH, PW, 3), 30, np.uint8), "changemap: waiting (freeze after warmup)", (120, 120, 255))
    for i, avi_idx in enumerate(warm_idx):
        ok, f = cap.read()
        if not ok:
            break
        knn.apply(cv2.resize(f, None, fx=S, fy=S)); n_warm += 1
        if i % show_every == 0 or i == len(warm_idx) - 1:
            bg_p = label(panel(knn.getBackgroundImage()), f"background (building {n_warm}/{len(warm_idx)})", (0, 255, 0))
            in_p = label(panel(f), f"{warm_kind} .avi frame idx={avi_idx}", (0, 255, 255))
            writer.write(compose(bg_p, in_p, waiting.copy(),
                                 f"{clabel}   |   PHASE 1 PRE-ROLL warmup {n_warm}/{len(warm_idx)}"))
    cap.release()

    frozen_bg_p = label(panel(knn.getBackgroundImage()), "FROZEN background (from pre-roll)", (0, 255, 0))
    for _ in range(int(FPS * 1.5)):
        writer.write(compose(frozen_bg_p,
                             label(panel(knn.getBackgroundImage()), "-> now freeze, query dynamic window", (0, 255, 0)),
                             label(np.full((PH, PW, 3), 30, np.uint8), "changemap: about to generate", (120, 120, 255)),
                             f"{clabel}   |   FREEZE background"))

    # PHASE 2 dynamic
    for j, name in enumerate(names, 1):
        img = cv2.imread(str(V18 / split / name))
        H, W = img.shape[:2]
        cm = cv2.resize(rm5(knn.apply(cv2.resize(img, None, fx=S, fy=S), learningRate=0)),
                        (W, H), interpolation=cv2.INTER_NEAREST)
        disp = cv2.resize(img, (PW, PH)); sx, sy = PW / W, PH / H
        for x, y, w, h in gt.get(name, []):
            cv2.rectangle(disp, (int(x*sx), int(y*sy)), (int((x+w)*sx), int((y+h)*sy)), (0, 255, 0), 2)
        writer.write(compose(frozen_bg_p.copy(),
                             label(disp, "RGB + GT (litter/garbage)", (0, 255, 0)),
                             label(cmap_panel(cm), "changemap (frozen-bg foreground)", (255, 255, 255)),
                             f"{clabel}   |   PHASE 2 DYNAMIC frame {j}/{len(names)} (in-clip f{frs[j-1]})"))
    writer.release()
    return "done", outp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="三 split 全 122 部")
    ap.add_argument("--split")
    ap.add_argument("--clip", help="ts_MMmSSs，例 20240605180713_00m20s")
    ap.add_argument("--overwrite", action="store_true")
    # 實體寫 HDD；experiments/.../資料前處理/preroll_montage_v18 有 symlink 指來此供瀏覽。
    ap.add_argument("--out", default="/path/to/data/preroll_montage_v18")
    a = ap.parse_args()

    if a.all:
        keys = sorted(ALIGN.keys())
    else:
        assert a.split and a.clip, "單片模式需 --split 與 --clip"
        keys = [f"{a.split}/{a.clip}"]

    cache = {}
    done = skip = 0
    for n, key in enumerate(keys, 1):
        split = ALIGN[key]["split"]
        if split not in cache:
            cache[split] = load_split(split)
        gt, names_by = cache[split]
        ck = key.split("/", 1)[1]
        names = names_by.get(ck)
        if not names:
            print(f"[{n}/{len(keys)}] ⚠️ 找不到 {key} 的 dataset 幀，跳過"); continue
        status, outp = make_one(key, gt, names, a.out, a.overwrite)
        done += status == "done"; skip += status == "skip"
        print(f"[{n}/{len(keys)}] {status:4s} {key}  ({len(names)}幀) → {outp.name}", flush=True)
    print(f"\n✅ 完成：新產 {done} 部、跳過(已存) {skip} 部，共 {len(keys)}。輸出 {a.out}/{{split}}/")

if __name__ == "__main__":
    main()
