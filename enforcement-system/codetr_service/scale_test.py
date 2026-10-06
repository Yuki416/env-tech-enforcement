"""CM_SCALE=0.5 vs 1.0：半解析度是不是讓最小的垃圾消失了？

完全照 gen_changemaps_v18.py 的配方重現：
  pass1 暖機 = 原始 .avi 的 [warm_start, first_abs)（事件前 pre-roll）
  pass2      = 對 dataset jpg 每張 knn.apply(learningRate=0)
唯一變因是 S（縮放比例）。

先驗證 S=0.5 能重現資料集裡既有的 changemap，重現得出來，S=1.0 的比較才有意義。
"""
import json, re, sys, collections
from pathlib import Path
import cv2, numpy as np

V18   = Path("/path/to/data/coco/chiayi_v18_own_dynamic_bgs_122vids")
ALIGN = "/workspace/project/experiments/18th_v18_own_dynamic_bgs_122vids/資料前處理/alignment_v18.json"
HIST, DIST, RM = 500, 500.0, 5
KEY  = sys.argv[1] if len(sys.argv) > 1 else "20240911175753_04m39s"
SPLIT = "train"

def rm_small(mask, min_area):
    fg = (mask == 255).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    out = np.zeros_like(fg)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= min_area:
            out[lab == i] = 1
    return (out * 255).astype(np.uint8)

def largest_blob(mask255, box, scale):
    """mask255 是在 scale 尺度下的遮罩；box 是全解析度 xywh。"""
    x, y, w, h = box
    sx, sy = int(x * scale), int(y * scale)
    sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
    H, W = mask255.shape[:2]
    x0, y0 = max(0, sx), max(0, sy)
    x1, y1 = min(W, sx + sw), min(H, sy + sh)
    if x1 <= x0 or y1 <= y0:
        return 0
    sub = (mask255[y0:y1, x0:x1] == 255).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(sub, 8)
    return int(max((st[i, cv2.CC_STAT_AREA] for i in range(1, n)), default=0))

def build_knn(avi, rng_):
    knn = cv2.createBackgroundSubtractorKNN(history=HIST, dist2Threshold=DIST, detectShadows=True)
    cap = cv2.VideoCapture(avi); n = 0
    if len(rng_) > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, rng_.start)
        for _ in rng_:
            ok, f = cap.read()
            if not ok: break
            knn.apply(f); n += 1          # 注意：暖機幀也要跟 pass2 同尺度，外面先 resize
    cap.release(); return knn, n

def run(scale, avi, warm, names, gt, min_area):
    knn = cv2.createBackgroundSubtractorKNN(history=HIST, dist2Threshold=DIST, detectShadows=True)
    cap = cv2.VideoCapture(avi); n = 0
    if len(warm) > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, warm.start)
        for _ in warm:
            ok, f = cap.read()
            if not ok: break
            knn.apply(cv2.resize(f, None, fx=scale, fy=scale) if scale != 1.0 else f); n += 1
    cap.release()
    out = {}
    for name in names:
        img = cv2.imread(str(V18 / SPLIT / name))
        small = cv2.resize(img, None, fx=scale, fy=scale) if scale != 1.0 else img
        fg = knn.apply(small, learningRate=0)
        cm = rm_small(fg, min_area)
        out[name] = [largest_blob(cm, b, scale) for _, b in gt[name]]
    return out, n

align = json.load(open(ALIGN))[f"{SPLIT}/{KEY}"]
ann = json.load(open(V18 / f"annotations/{SPLIT}.json"))
id2 = {im["id"]: im["file_name"] for im in ann["images"] if KEY in im["file_name"]}
gt = collections.defaultdict(list)
for a in ann["annotations"]:
    if a["image_id"] in id2 and a["category_id"] in (2, 5):
        gt[id2[a["image_id"]]].append(("litter" if a["category_id"] == 2 else "garbage", a["bbox"]))
names = sorted(gt, key=lambda n: int(re.search(r"_frame_(\d+)", n).group(1)))
warm = range(align["warm_start"], align["first_abs"])
print(f"clip {KEY}   avi 暖機範圍 [{warm.start}, {warm.stop}) = {len(warm)} 幀   {len(names)} 張有標註\n")

half, nw = run(0.5, align["avi"], warm, names, gt, RM)
full, _  = run(1.0, align["avi"], warm, names, gt, RM)          # 門檻沿用 5
full20,_ = run(1.0, align["avi"], warm, names, gt, RM * 4)      # 門檻等比放大到 20

# 資料集既存的 changemap（半解析度還原）當基準
stored = {}
for name in names:
    cm = cv2.imread(str(V18 / f"{SPLIT}_changemap" / (name[:-4] + ".png")), cv2.IMREAD_GRAYSCALE)
    H, W = cm.shape[:2]
    h2 = cv2.resize(cm, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
    stored[name] = [largest_blob(h2, b, 0.5) for _, b in gt[name]]

print(f"{'frame':>6} {'框 w×h':>10} {'資料集既有':>10} {'重現 S=0.5':>11} "
      f"{'S=1.0 門檻5':>12} {'S=1.0 門檻20':>13} {'折算半解析':>11}")
print("-" * 82)
agree = tot = 0
for name in names:
    fn = int(re.search(r"_frame_(\d+)", name).group(1))
    for i, (lab, (x, y, w, h)) in enumerate(gt[name]):
        st_, hf, fl, f20 = stored[name][i], half[name][i], full[name][i], full20[name][i]
        tot += 1
        if st_ == hf: agree += 1
        print(f"{fn:>6} {int(w)}×{int(h):<7} {st_:>10} {hf:>11} {fl:>12} {f20:>13} {fl/4:>10.1f}")
print("-" * 82)
print(f"重現一致：{agree}/{tot} 個框與資料集既有 changemap 完全相同\n")

for tag, d in (("S=0.5 門檻5", half), ("S=1.0 門檻5", full), ("S=1.0 門檻20", full20)):
    v = np.array([x for name in names for x in d[name]])
    print(f"{tag:<14} blob=0 有 {(v==0).sum():>2} 個 / {len(v)}   中位數 {np.median(v):>6.1f}")
