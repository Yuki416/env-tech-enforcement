"""暖機來源 A/B：訓練式（事件前 pre-roll）vs 線上式（影片開頭 150 幀）。

前一次的 parity 只量了 changemap 本身（遮罩 IoU、GT 框內前景比），
**沒有量「那個差異會不會讓模型抓到不一樣的東西」**。這支就是補那個缺口。

作法：同一支 .avi、同一批幀、同一顆模型、同一條推論路徑
（直接 import codetr_service.server 的 Detector，確保跟正式服務一模一樣），
唯一變因是 changemap 的暖機範圍。

輸出三組數字：
  1. 每個版本各自對 GT 的 TP / FP / FN（score≥0.3、IoU≥0.5、同類）
  2. 兩版本之間的一致率（同一幀、同類、IoU≥0.5 視為同一個偵測）
  3. 分數差（配得起來的偵測，score 差多少）

用法（在掛好 mmdet 的 co-detr 容器內）：
    python /app/codetr_service/ab_warmup_effect.py [--splits test] [--limit-clips N]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/app")
from codetr_service.server import (  # noqa: E402
    CM_DIST2,
    CM_HISTORY,
    CM_SCALE,
    CM_WARM_FRAMES,
    Detector,
    rm5,
)

ALIGN = "/workspace/project/experiments/18th_v18_own_dynamic_bgs_122vids/資料前處理/alignment_v18.json"
V18 = Path("/path/to/data/coco/chiayi_v18_own_dynamic_bgs_122vids")
SCORE_THR = 0.30
IOU_THR = 0.50
TARGET = {"litter", "garbage"}
CAT_NAME = {1: "car", 2: "litter", 3: "motorcycle", 4: "person", 5: "garbage"}


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def build_knn(avi: str, warm_range: range):
    knn = cv2.createBackgroundSubtractorKNN(
        history=CM_HISTORY, dist2Threshold=CM_DIST2, detectShadows=True
    )
    cap = cv2.VideoCapture(avi)
    n = 0
    if len(warm_range) > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, warm_range.start)
        for _ in warm_range:
            ok, f = cap.read()
            if not ok:
                break
            knn.apply(cv2.resize(f, None, fx=CM_SCALE, fy=CM_SCALE))
            n += 1
    cap.release()
    return knn, n


def changemap(knn, frame):
    h, w = frame.shape[:2]
    fg = knn.apply(cv2.resize(frame, None, fx=CM_SCALE, fy=CM_SCALE), learningRate=0)
    return cv2.resize(rm5(fg), (w, h), interpolation=cv2.INTER_NEAREST)


def match_to_gt(preds, gts):
    """回傳 (tp, fp, fn)。每個 GT 最多配一個偵測，貪婪取分數高的先配。"""
    used = set()
    tp = 0
    for p in sorted(preds, key=lambda d: -d["score"]):
        best, best_iou = None, IOU_THR
        for gi, g in enumerate(gts):
            if gi in used or g["label"] != p["label"]:
                continue
            v = iou(p["box"], g["box"])
            if v >= best_iou:
                best, best_iou = gi, v
        if best is not None:
            used.add(best)
            tp += 1
    return tp, len(preds) - tp, len(gts) - tp


def agreement(pa, pb):
    """兩版本之間配得起來的偵測數，以及配對的分數差。"""
    used = set()
    matched, dscores = 0, []
    for p in sorted(pa, key=lambda d: -d["score"]):
        best, best_iou = None, IOU_THR
        for bi, q in enumerate(pb):
            if bi in used or q["label"] != p["label"]:
                continue
            v = iou(p["box"], q["box"])
            if v >= best_iou:
                best, best_iou = bi, v
        if best is not None:
            used.add(best)
            matched += 1
            dscores.append(pb[best]["score"] - p["score"])
    return matched, dscores


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="*", default=["test"])
    ap.add_argument("--limit-clips", type=int, default=0)
    args = ap.parse_args()

    align = json.load(open(ALIGN))
    det = Detector(None, "cuda:0")
    print(f"模型：{Path(det.checkpoint).name}\n", flush=True)

    agg = collections.defaultdict(lambda: collections.Counter())
    all_dscores: list[float] = []
    n_frames = 0

    for split in args.splits:
        ann = json.load(open(V18 / f"annotations/{split}.json"))
        id2fn = {im["id"]: im["file_name"] for im in ann["images"]}
        gt_by_fn: dict[str, list] = collections.defaultdict(list)
        for a in ann["annotations"]:
            name = CAT_NAME.get(a["category_id"], "")
            if name in TARGET:
                x, y, w, h = a["bbox"]
                gt_by_fn[id2fn[a["image_id"]]].append(
                    {"label": name, "box": [x, y, x + w, y + h]}
                )

        clips: dict[str, list[str]] = collections.defaultdict(list)
        for im in ann["images"]:
            g = re.search(r"(\d{14})_(?:(\d{2})h)?(\d{2})m(\d{2})s", im["file_name"])
            offs = (int(g.group(2)) if g.group(2) else 0) * 3600 \
                + int(g.group(3)) * 60 + int(g.group(4))
            clips[f"{split}/{g.group(1)}_{offs//60:02d}m{offs%60:02d}s"].append(
                im["file_name"]
            )

        keys = sorted(clips)
        if args.limit_clips:
            keys = keys[: args.limit_clips]

        print(f"{'clip':<24} {'幀':>4} "
              f"{'訓練 TP/FP/FN':>14} {'線上 TP/FP/FN':>14} {'一致':>10}", flush=True)
        print("-" * 74, flush=True)

        for key in keys:
            info = align.get(key)
            if info is None or not Path(info["avi"]).is_file():
                continue
            knn_tr, n_tr = build_knn(info["avi"], range(info["warm_start"], info["first_abs"]))
            knn_on, n_on = build_knn(info["avi"], range(0, CM_WARM_FRAMES))
            if n_tr == 0 or n_on == 0:
                continue

            c = collections.Counter()
            cap = cv2.VideoCapture(info["avi"])
            for name in sorted(clips[key]):
                fn = int(re.search(r"_frame_(\d+)", name).group(1))
                cap.set(cv2.CAP_PROP_POS_FRAMES, info["first_abs"] + fn)
                ok, frame = cap.read()
                if not ok:
                    continue
                n_frames += 1
                gts = gt_by_fn.get(name, [])

                preds = {}
                for tag, knn in (("tr", knn_tr), ("on", knn_on)):
                    raw = det.detect(frame, changemap(knn, frame))
                    preds[tag] = [
                        {"label": b["label"],
                         "box": [b["x1"], b["y1"], b["x2"], b["y2"]],
                         "score": b["score"]}
                        for b in raw
                        if b["score"] >= SCORE_THR and b["label"] in TARGET
                    ]

                for tag in ("tr", "on"):
                    tp, fp, fn_ = match_to_gt(preds[tag], gts)
                    c[f"{tag}_tp"] += tp
                    c[f"{tag}_fp"] += fp
                    c[f"{tag}_fn"] += fn_
                    c[f"{tag}_det"] += len(preds[tag])
                m, ds = agreement(preds["tr"], preds["on"])
                c["match"] += m
                all_dscores.extend(ds)
            cap.release()

            for k, v in c.items():
                agg["total"][k] += v
            denom = max(c["tr_det"], c["on_det"], 1)
            print(f"{key[len(split)+1:]:<24} {len(clips[key]):>4} "
                  f"{c['tr_tp']:>4}/{c['tr_fp']:>4}/{c['tr_fn']:>4} "
                  f"{c['on_tp']:>5}/{c['on_fp']:>4}/{c['on_fn']:>4} "
                  f"{c['match']}/{denom:>7}", flush=True)

    t = agg["total"]
    print("-" * 74)
    print(f"\n總計（{n_frames} 幀，score≥{SCORE_THR}、IoU≥{IOU_THR}）\n")
    print(f"{'版本':<14} {'偵測數':>7} {'TP':>5} {'FP':>5} {'FN':>5} "
          f"{'Precision':>10} {'Recall':>8}")
    print("-" * 62)
    for tag, label in (("tr", "訓練式暖機"), ("on", "線上式暖機")):
        tp, fp, fn_ = t[f"{tag}_tp"], t[f"{tag}_fp"], t[f"{tag}_fn"]
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn_) if tp + fn_ else 0.0
        print(f"{label:<14} {t[f'{tag}_det']:>7} {tp:>5} {fp:>5} {fn_:>5} "
              f"{p:>10.3f} {r:>8.3f}")
    print("-" * 62)
    denom = max(t["tr_det"], t["on_det"], 1)
    print(f"\n兩版本一致的偵測：{t['match']} / {denom} = {t['match']/denom:.1%}")
    if all_dscores:
        a = np.array(all_dscores)
        print(f"配對偵測的分數差（線上 − 訓練）：平均 {a.mean():+.3f}、"
              f"中位數 {np.median(a):+.3f}、絕對值平均 {np.abs(a).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
