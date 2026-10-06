"""量垃圾在畫面裡到底有多大 —— 兩個尺度：

1. **GT 框面積**（全解析度）：標註者框出來的大小，是上界。
2. **changemap 裡實際亮起來的 blob 面積**（半解析度）：`rm5()` 真正看到的東西。
   dataset 的 changemap 是「半解析度算完 → INTER_NEAREST 放大」，
   所以先縮回半解析度還原成 rm5 當時看到的樣子，再跑連通元件。

回答的問題：CM_MIN_AREA=5 這個門檻，會不會把真的垃圾整個清掉？
"""
import collections, json, sys
from pathlib import Path
import cv2, numpy as np

V18 = Path("/path/to/data/coco/chiayi_v18_own_dynamic_bgs_122vids")
CAT = {2: "litter", 5: "garbage"}
THRESHOLDS = [1, 3, 5, 10, 20, 50, 100]

def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")

def main():
    splits = sys.argv[1:] or ["train", "valid", "test"]
    bbox_area = collections.defaultdict(list)   # 全解析度 px
    blob_area = collections.defaultdict(list)   # 半解析度 px（rm5 的尺度）
    res_seen  = collections.Counter()
    n_img = n_missing_cm = 0

    for split in splits:
        ann = json.load(open(V18 / f"annotations/{split}.json"))
        id2fn = {im["id"]: im["file_name"] for im in ann["images"]}
        by_img = collections.defaultdict(list)
        for a in ann["annotations"]:
            if a["category_id"] in CAT:
                by_img[id2fn[a["image_id"]]].append((CAT[a["category_id"]], a["bbox"]))

        for name, anns in by_img.items():
            cm_path = V18 / f"{split}_changemap" / (name[:-4] + ".png")
            cm = cv2.imread(str(cm_path), cv2.IMREAD_GRAYSCALE)
            if cm is None:
                n_missing_cm += 1
                cm_half = None
            else:
                H, W = cm.shape[:2]
                res_seen[f"{W}x{H}"] += 1
                # 還原成 rm5 當時看到的半解析度遮罩
                cm_half = cv2.resize(cm, (W // 2, H // 2), interpolation=cv2.INTER_NEAREST)
            n_img += 1

            for label, (x, y, w, h) in anns:
                bbox_area[label].append(w * h)
                if cm_half is None:
                    continue
                hx, hy, hw, hh = int(x / 2), int(y / 2), max(1, int(w / 2)), max(1, int(h / 2))
                Hh, Wh = cm_half.shape[:2]
                x0, y0 = max(0, hx), max(0, hy)
                x1, y1 = min(Wh, hx + hw), min(Hh, hy + hh)
                if x1 <= x0 or y1 <= y0:
                    continue
                sub = (cm_half[y0:y1, x0:x1] == 255).astype(np.uint8)
                nlab, lab, st, _ = cv2.connectedComponentsWithStats(sub, 8)
                largest = max((st[i, cv2.CC_STAT_AREA] for i in range(1, nlab)), default=0)
                blob_area[label].append(int(largest))

    print(f"影像 {n_img} 張（缺 changemap {n_missing_cm} 張）；解析度 {dict(res_seen)}\n")

    print("── 一、GT 框面積（全解析度 px）──")
    print(f"{'類別':<9} {'框數':>6} {'最小':>8} {'p5':>8} {'中位數':>9} {'p95':>9} {'最大':>9} {'中位數邊長':>11}")
    for lab in ("litter", "garbage"):
        a = np.array(bbox_area[lab])
        if not len(a): continue
        print(f"{lab:<9} {len(a):>6} {a.min():>8.0f} {pct(a,5):>8.0f} {pct(a,50):>9.0f} "
              f"{pct(a,95):>9.0f} {a.max():>9.0f} {np.sqrt(pct(a,50)):>10.0f}px")

    print("\n── 二、changemap 實際 blob 面積（半解析度 px ＝ rm5 看到的尺度）──")
    print(f"{'類別':<9} {'框數':>6} {'最小':>8} {'p5':>8} {'p25':>8} {'中位數':>9} {'p95':>9} {'最大':>9}")
    for lab in ("litter", "garbage"):
        a = np.array(blob_area[lab])
        if not len(a): continue
        print(f"{lab:<9} {len(a):>6} {a.min():>8.0f} {pct(a,5):>8.0f} {pct(a,25):>8.0f} "
              f"{pct(a,50):>9.0f} {pct(a,95):>9.0f} {a.max():>9.0f}")

    print("\n── 三、各門檻會清掉多少比例的真垃圾（blob < 門檻）──")
    print(f"{'門檻':>6}", end="")
    for lab in ("litter", "garbage"):
        print(f" {lab+' 被清掉':>16}", end="")
    print()
    for t in THRESHOLDS:
        print(f"{t:>6}", end="")
        for lab in ("litter", "garbage"):
            a = np.array(blob_area[lab])
            r = float((a < t).mean()) * 100 if len(a) else float("nan")
            print(f" {r:>14.1f}% ", end="")
        print(f"   {'← 目前設定' if t == 5 else ''}")

if __name__ == "__main__":
    main()
