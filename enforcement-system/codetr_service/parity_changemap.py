"""線上 changemap（暖機用影片開頭）vs 訓練時 changemap（暖機用事件前 pre-roll）的 parity 比對。

兩者唯一差別是 pass1 暖機餵哪些幀；pass2 一律 learningRate=0、同一組後處理、同一批輸入幀。
輸入幀統一取自 .avi（不是 dataset jpg），這樣就把「jpg 重編碼」這個變因也排除掉。

關鍵指標是 **GT 框內前景比**：changemap 的用途是告訴模型「這裡有東西在動」，
如果線上版在 litter/garbage 框內仍然點得亮，暖機來源的差異就不影響模型判讀。
"""
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ALIGN = "/workspace/project/experiments/18th_v18_own_dynamic_bgs_122vids/資料前處理/alignment_v18.json"
V18 = Path("/workspace/datasets/coco/chiayi_v18_own_dynamic_bgs_122vids")
S, HIST, DIST, RM = 0.5, 500, 500.0, 5
WARM = 150
LITTER, GARBAGE = 2, 5


def rm5(mask):
    fg = (mask == 255).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    out = np.zeros_like(fg)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= RM:
            out[lab == i] = 1
    return (out * 255).astype(np.uint8)


def build_knn(avi, warm_range):
    knn = cv2.createBackgroundSubtractorKNN(history=HIST, dist2Threshold=DIST, detectShadows=True)
    cap = cv2.VideoCapture(avi)
    n = 0
    if len(warm_range) > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, warm_range.start)
        for _ in warm_range:
            ok, f = cap.read()
            if not ok:
                break
            knn.apply(cv2.resize(f, None, fx=S, fy=S))
            n += 1
    cap.release()
    return knn, n


def apply_frozen(knn, frame):
    h, w = frame.shape[:2]
    fg = knn.apply(cv2.resize(frame, None, fx=S, fy=S), learningRate=0)
    return cv2.resize(rm5(fg), (w, h), interpolation=cv2.INTER_NEAREST)


def box_fg(cm, bbox):
    x, y, w, h = [int(round(v)) for v in bbox]
    H, W = cm.shape[:2]
    x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return float((cm[y0:y1, x0:x1] > 0).mean())


def main():
    align = json.load(open(ALIGN))
    ann = json.load(open(V18 / "annotations/test.json"))
    id2fn = {im["id"]: im["file_name"] for im in ann["images"]}
    gt = {}
    for a in ann["annotations"]:
        if a["category_id"] in (LITTER, GARBAGE):
            gt.setdefault(id2fn[a["image_id"]], []).append(a["bbox"])

    # 每個 test clip 的 dataset 幀
    clips = {}
    for im in ann["images"]:
        g = re.search(r"(\d{14})_(?:(\d{2})h)?(\d{2})m(\d{2})s", im["file_name"])
        offs = (int(g.group(2)) if g.group(2) else 0) * 3600 + int(g.group(3)) * 60 + int(g.group(4))
        clips.setdefault(f"test/{g.group(1)}_{offs//60:02d}m{offs%60:02d}s", []).append(im["file_name"])

    print(f"{'clip':<28} {'first_abs':>9} {'IoU':>6} {'訓練框內':>8} {'線上框內':>8} {'框數':>4}", flush=True)
    print("-" * 72, flush=True)
    rows = []
    skipped = 0
    for key in sorted(clips):
        info = align.get(key)
        if info is None or not Path(info["avi"]).is_file():
            skipped += 1
            print(f"  [skip] {key}  avi={info['avi'] if info else 'NO-ALIGN'}", flush=True)
            continue
        names = sorted(clips[key])
        knn_train, n_tr = build_knn(info["avi"], range(info["warm_start"], info["first_abs"]))
        knn_online, n_on = build_knn(info["avi"], range(0, WARM))
        if n_tr == 0 or n_on == 0:
            continue

        cap = cv2.VideoCapture(info["avi"])
        inter = union = 0
        tr_ratios, on_ratios, nbox = [], [], 0
        for name in names:
            fn = int(re.search(r"_frame_(\d+)", name).group(1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, info["first_abs"] + fn)
            ok, frame = cap.read()
            if not ok:
                continue
            cm_tr = apply_frozen(knn_train, frame)
            cm_on = apply_frozen(knn_online, frame)
            a, b = cm_tr > 0, cm_on > 0
            inter += int((a & b).sum())
            union += int((a | b).sum())
            for bbox in gt.get(name, []):
                # dataset jpg 與 avi 幀解析度一致（前處理沒縮放），bbox 直接用
                r_tr, r_on = box_fg(cm_tr, bbox), box_fg(cm_on, bbox)
                if r_tr is not None and r_on is not None:
                    tr_ratios.append(r_tr)
                    on_ratios.append(r_on)
                    nbox += 1
        cap.release()

        iou = inter / union if union else float("nan")
        mt = float(np.mean(tr_ratios)) if tr_ratios else float("nan")
        mo = float(np.mean(on_ratios)) if on_ratios else float("nan")
        print(f"{key[5:]:<28} {info['first_abs']:>9} {iou:>6.3f} {mt:>8.3f} {mo:>8.3f} {nbox:>4}")
        rows.append((key, iou, mt, mo, nbox, info["first_abs"]))

    if rows:
        print("-" * 72)
        ious = [r[1] for r in rows if not np.isnan(r[1])]
        tr = [r[2] for r in rows if not np.isnan(r[2])]
        on = [r[3] for r in rows if not np.isnan(r[3])]
        print(f"{'平均':<28} {'':>9} {np.mean(ious):>6.3f} {np.mean(tr):>8.3f} {np.mean(on):>8.3f} {sum(r[4] for r in rows):>4}")


if __name__ == "__main__":
    sys.exit(main())
