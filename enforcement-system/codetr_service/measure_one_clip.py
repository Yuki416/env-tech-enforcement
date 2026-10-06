"""單一 clip 的垃圾尺寸明細。方法同 measure_litter_size.py：
changemap 用 INTER_NEAREST 縮回半解析度，還原成 rm5() 當時看到的遮罩。"""
import json, re, sys, collections
from pathlib import Path
import cv2, numpy as np

V18 = Path("/path/to/data/coco/chiayi_v18_own_dynamic_bgs_122vids")
CAT = {2: "litter", 5: "garbage"}
KEY = sys.argv[1] if len(sys.argv) > 1 else "20240911175753_04m39s"

for split in ("train", "valid", "test"):
    ann = json.load(open(V18 / f"annotations/{split}.json"))
    imgs = [im for im in ann["images"] if KEY in im["file_name"]]
    if not imgs:
        continue
    id2 = {im["id"]: im for im in imgs}
    per = collections.defaultdict(list)
    for a in ann["annotations"]:
        if a["image_id"] in id2 and a["category_id"] in CAT:
            per[a["image_id"]].append((CAT[a["category_id"]], a["bbox"]))

    print(f"split = {split}   clip = {KEY}")
    print(f"影像 {len(imgs)} 張   有垃圾標註的 {len(per)} 張")
    print(f"檔名樣式：{imgs[0]['file_name']}")
    print(f"解析度：{imgs[0]['width']}x{imgs[0]['height']}\n")

    print(f"{'frame':>6} {'類別':<8} {'框 w×h':>12} {'框面積':>9} {'blob(半解析)':>12} {'佔框比':>8} {'≥5?':>5}")
    print("-" * 70)
    rows = []
    for im in sorted(imgs, key=lambda i: int(re.search(r"_frame_(\d+)", i["file_name"]).group(1))):
        fn = int(re.search(r"_frame_(\d+)", im["file_name"]).group(1))
        anns = per.get(im["id"], [])
        if not anns:
            continue
        cm = cv2.imread(str(V18 / f"{split}_changemap" / (im["file_name"][:-4] + ".png")),
                        cv2.IMREAD_GRAYSCALE)
        H, W = cm.shape[:2]
        half = cv2.resize(cm, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
        for label, (x, y, w, h) in anns:
            hx, hy = int(x / 2), int(y / 2)
            hw, hh = max(1, int(w / 2)), max(1, int(h / 2))
            Hh, Wh = half.shape[:2]
            x0, y0 = max(0, hx), max(0, hy)
            x1, y1 = min(Wh, hx + hw), min(Hh, hy + hh)
            sub = (half[y0:y1, x0:x1] == 255).astype(np.uint8)
            nlab, lab, st, _ = cv2.connectedComponentsWithStats(sub, 8)
            largest = max((st[i, cv2.CC_STAT_AREA] for i in range(1, nlab)), default=0)
            box_half = max(1, hw * hh)
            ratio = largest / box_half * 100
            rows.append((label, w * h, largest))
            print(f"{fn:>6} {label:<8} {int(w)}×{int(h):<8} {int(w*h):>9} "
                  f"{largest:>12} {ratio:>7.1f}% {'✓' if largest >= 5 else '✗ 會被清掉':>5}")

    print("-" * 70)
    for lab in ("litter", "garbage"):
        a = np.array([r[1] for r in rows if r[0] == lab])
        b = np.array([r[2] for r in rows if r[0] == lab])
        if not len(a):
            continue
        print(f"{lab}: {len(a)} 個框   框面積 中位 {np.median(a):.0f} "
              f"(邊長約 {np.sqrt(np.median(a)):.0f}px)   blob 中位 {np.median(b):.0f}   "
              f"blob=0 有 {(b==0).sum()} 個   blob<5 有 {(b<5).sum()} 個")
