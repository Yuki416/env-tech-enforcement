#!/usr/bin/env python3
"""
Co-DETR v9 訓練後分析腳本

四個模式，訓練結束後依序執行：

  1. ap-curve       — 從 training_log.md 產出 litter/garbage AP 曲線報告
  2. by-scene       — 從 per_frame_class_errors CSV 產出場景分組分析
  3. score-dist     — 對 test set 跑低閾值推論，產出 score 分布報告
  4. comparison     — 對比 v8 vs v9 的整體結果

執行範例（容器內）：
  python /workspace/project/post_analysis_v9.py --mode ap-curve \
    --v9-log  /workspace/project/results/chiayi_v9_36e/training_log.md \
    --output-dir /workspace/project/results/chiayi_v9_36e/error_analysis

  python /workspace/project/post_analysis_v9.py --mode by-scene \
    --error-dir /workspace/project/results/chiayi_v9_36e/error_analysis \
    --output-dir /workspace/project/results/chiayi_v9_36e/error_analysis

  python /workspace/project/post_analysis_v9.py --mode score-dist \
    --config     /workspace/project/configs/co_dino_chiayi_v9_36e.py \
    --checkpoint /workspace/project/results/chiayi_v9_36e/best_coco_bbox_mAP_epoch_X.pth \
    --data-root  /workspace/datasets/coco/chiayi_v9_garbage_litter \
    --output-dir /workspace/project/results/chiayi_v9_36e/error_analysis

  python /workspace/project/post_analysis_v9.py --mode comparison \
    --v8-log      /workspace/project/results/chiayi_v8_dense_36e/training_log.md \
    --v9-log      /workspace/project/results/chiayi_v9_36e/training_log.md \
    --v8-error-dir /workspace/project/results/chiayi_v8_dense_36e/error_analysis \
    --v9-error-dir /workspace/project/results/chiayi_v9_36e/error_analysis \
    --output-dir  /workspace/project/results/chiayi_v9_36e/error_analysis
"""

import os
import re
import csv
import json
import argparse
from collections import defaultdict
from datetime import datetime

import numpy as np

# ── 常數 ─────────────────────────────────────────────────────────────────────

V9_CLASSES   = ["car", "litter", "motorcycle", "person", "garbage"]
V8_CLASSES   = ["car", "litter", "motorcycle", "person"]
SPLITS       = ["train", "valid", "test"]
SCORE_BINS   = [(round(i * 0.1, 1), round((i + 1) * 0.1, 1)) for i in range(10)]
IOU_THR      = 0.5
SCORE_DIST_THR = 0.01   # 低閾值，捕捉所有預測


# ── training_log.md 解析 ──────────────────────────────────────────────────────

def parse_mmengine_log(log_path: str, classes: list) -> list:
    """解析 MMEngine 原始 log，從 Epoch(val) 行提取每個 epoch 的驗證指標。
    只保留含 coco/bbox_mAP 的行（最終驗證行），每個 epoch 取最後一筆。"""
    pattern = re.compile(r'Epoch\(val\) \[(\d+)\]\[\d+/\d+\]\s+(.*)')
    metric_pattern = re.compile(r'(coco/[\w_]+): ([\d.]+)')
    best: dict = {}
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            if "coco/bbox_mAP:" not in line:
                continue
            epoch_num = int(m.group(1))
            metrics = {k: float(v) for k, v in metric_pattern.findall(m.group(2))}
            row = {
                "epoch": epoch_num,
                "lr":    1.25e-5,
                "mAP":   metrics.get("coco/bbox_mAP"),
            }
            for cls in classes:
                row[f"{cls}_ap"] = metrics.get(f"coco/{cls}_precision")
            best[epoch_num] = row
    return sorted(best.values(), key=lambda e: e["epoch"])


def parse_training_log(log_path: str, classes: list) -> list:
    """
    解析 training_log.md，回傳 list of dict（每個 epoch 一筆）：
    {"epoch", "lr", "mAP", "car_ap", "litter_ap", "motorcycle_ap",
     "person_ap", "garbage_ap"（v9 才有）}
    """
    with open(log_path, encoding="utf-8") as f:
        content = f.read()

    sections = re.split(r'\n## Epoch (\d+)\n', content)
    epochs = []

    for i in range(1, len(sections), 2):
        epoch_num = int(sections[i])
        body = sections[i + 1]

        lr_m = re.search(r'\*\*LR:\*\*\s*([\d.e+\-]+)', body)
        lr = float(lr_m.group(1)) if lr_m else None

        metrics = {}
        for line in body.split('\n'):
            m = re.match(r'\|\s*coco/([\w_]+)\s*\|\s*([\d.]+)\s*\|', line)
            if m:
                metrics[m.group(1)] = float(m.group(2))

        row = {
            "epoch": epoch_num,
            "lr":    lr,
            "mAP":   metrics.get("bbox_mAP"),
        }
        for cls in classes:
            row[f"{cls}_ap"] = metrics.get(f"{cls}_precision")

        epochs.append(row)

    return epochs


# ── Mode 1: ap-curve ──────────────────────────────────────────────────────────

def mode_ap_curve(args):
    print("=== ap-curve：litter/garbage AP 曲線 ===")
    epochs = parse_training_log(args.v9_log, V9_CLASSES)

    if not epochs:
        print("ERROR: 無法解析 training_log.md，請確認路徑正確。")
        return

    if getattr(args, "v9_log_extra", None):
        extra = parse_mmengine_log(args.v9_log_extra, V9_CLASSES)
        extra_eps = {e["epoch"] for e in extra}
        epochs = sorted(extra + [e for e in epochs if e["epoch"] not in extra_eps],
                        key=lambda e: e["epoch"])

    # LR decay 位置
    decay_epochs = []
    for i in range(1, len(epochs)):
        if epochs[i]["lr"] and epochs[i - 1]["lr"]:
            if epochs[i]["lr"] < epochs[i - 1]["lr"] * 0.5:
                decay_epochs.append(epochs[i]["epoch"])

    out_path = os.path.join(args.output_dir, "litter_garbage_ap_curve.md")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Litter & Garbage AP 曲線分析\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write(f"- Config: chiayi_v9_36e\n")
        f.write(f"- LR decay epochs: {decay_epochs}\n\n")

        # 表格
        f.write("## Per-Epoch 指標\n\n")
        header = "| Epoch | LR | mAP | litter AP | garbage AP | car AP | motorcycle AP | person AP |\n"
        f.write(header)
        f.write("|-------|-----|-----|-----------|------------|--------|---------------|-----------|\n")
        for e in epochs:
            lr_str  = f"{e['lr']:.2e}" if e["lr"] else "—"
            mAP     = f"{e['mAP']:.4f}" if e["mAP"] is not None else "—"
            l_ap    = f"{e.get('litter_ap', 0) or 0:.4f}"
            g_ap    = f"{e.get('garbage_ap', 0) or 0:.4f}"
            car_ap  = f"{e.get('car_ap', 0) or 0:.4f}"
            moto_ap = f"{e.get('motorcycle_ap', 0) or 0:.4f}"
            pers_ap = f"{e.get('person_ap', 0) or 0:.4f}"
            decay_mark = " ⬇️LR" if e["epoch"] in decay_epochs else ""
            f.write(f"| {e['epoch']}{decay_mark} | {lr_str} | {mAP} | {l_ap} | {g_ap} "
                    f"| {car_ap} | {moto_ap} | {pers_ap} |\n")

        # 文字分析
        f.write("\n## 分析\n\n")

        litter_vals  = [(e["epoch"], e.get("litter_ap") or 0)  for e in epochs]
        garbage_vals = [(e["epoch"], e.get("garbage_ap") or 0) for e in epochs]

        def first_nonzero_epoch(vals, thr=0.01):
            for ep, v in vals:
                if v > thr:
                    return ep
            return None

        l_start = first_nonzero_epoch(litter_vals)
        g_start = first_nonzero_epoch(garbage_vals)
        f.write(f"- **litter** 首次超過 0.01 AP：Epoch {l_start or '未達到'}\n")
        f.write(f"- **garbage** 首次超過 0.01 AP：Epoch {g_start or '未達到'}\n\n")

        for dec_ep in decay_epochs:
            before = next((e for e in reversed(epochs) if e["epoch"] < dec_ep), None)
            after  = next((e for e in epochs if e["epoch"] >= dec_ep), None)
            if before and after:
                f.write(f"### LR Decay @ Epoch {dec_ep}\n\n")
                f.write(f"| 指標 | Epoch {before['epoch']} (decay 前) | Epoch {after['epoch']} (decay 後) |\n")
                f.write("|------|------|------|\n")
                for key, label in [("mAP","mAP"),("litter_ap","litter AP"),("garbage_ap","garbage AP")]:
                    bv = f"{before.get(key) or 0:.4f}"
                    av = f"{after.get(key) or 0:.4f}"
                    f.write(f"| {label} | {bv} | {av} |\n")
                f.write("\n")

        best_l  = max(litter_vals,  key=lambda x: x[1])
        best_g  = max(garbage_vals, key=lambda x: x[1])
        final_l = litter_vals[-1]  if litter_vals  else (0, 0)
        final_g = garbage_vals[-1] if garbage_vals else (0, 0)
        f.write(f"### Best / Final AP\n\n")
        f.write(f"| 類別 | Best AP | Best Epoch | Final AP (Ep {len(epochs)}) |\n")
        f.write("|------|---------|------------|----------------------------|\n")
        f.write(f"| litter  | {best_l[1]:.4f}  | {best_l[0]}  | {final_l[1]:.4f} |\n")
        f.write(f"| garbage | {best_g[1]:.4f} | {best_g[0]} | {final_g[1]:.4f} |\n\n")

        if best_l[1] > best_g[1]:
            f.write("- **litter** 整體學得較好（best AP 較高）。\n")
        elif best_g[1] > best_l[1]:
            f.write("- **garbage** 整體學得較好（best AP 較高）。\n")
        else:
            f.write("- litter 與 garbage 表現相近。\n")

        last_mAPs = [e["mAP"] for e in epochs[-5:] if e["mAP"] is not None]
        if last_mAPs and max(last_mAPs) - min(last_mAPs) < 0.01:
            f.write("- mAP 在最後 5 epoch 趨於穩定，未見明顯過擬合。\n")
        else:
            f.write("- mAP 在最後 5 epoch 仍有波動，可能需要更多 epoch 或已過擬合。\n")

    print(f"  寫入 {out_path}")


# ── Mode 2: by-scene ──────────────────────────────────────────────────────────

def mode_by_scene(args):
    print("=== by-scene：場景分組分析 ===")

    SCENE_PREFIXES = {
        "Day_Litter":   "litter",
        "Night_Litter": "litter",
        "Day_Garbage":  "garbage",
        "Night_Garbage":"garbage",
    }

    def get_scene(video_name: str):
        for prefix, cls in SCENE_PREFIXES.items():
            if video_name.startswith(prefix):
                return prefix, cls
        return "Other", None

    # {split: {scene: {class: {frames, TP, FN, FP}}}}
    all_data = {}

    for split in SPLITS:
        csv_path = os.path.join(args.error_dir, f"{split}_per_frame_class_errors.csv")
        if not os.path.exists(csv_path):
            print(f"  [SKIP] 找不到 {csv_path}")
            continue

        scene_stats = defaultdict(lambda: defaultdict(
            lambda: {"frames_with_gt": 0, "TP": 0, "FN": 0, "FP": 0}
        ))
        video_seen  = defaultdict(set)

        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video = row["video_name"]
                cls   = row["class"]
                if cls not in ("litter", "garbage"):
                    continue
                scene, target_cls = get_scene(video)
                if target_cls != cls:
                    continue

                tp = int(row["TP"]); fn = int(row["FN"]); fp = int(row["FP"])
                scene_stats[scene][cls]["TP"] += tp
                scene_stats[scene][cls]["FN"] += fn
                scene_stats[scene][cls]["FP"] += fp
                if tp + fn > 0:
                    scene_stats[scene][cls]["frames_with_gt"] += 1

        all_data[split] = scene_stats

    out_path = os.path.join(args.output_dir, "litter_garbage_by_scene.md")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Litter & Garbage 場景分組分析\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write("> 各場景統計的是「該場景目標類別」的 TP/FN/FP。\n")
        f.write("> Day_Litter / Night_Litter → 統計 litter；Day_Garbage / Night_Garbage → 統計 garbage。\n\n")

        for split in SPLITS:
            if split not in all_data:
                continue
            f.write(f"---\n\n## {split.upper()}\n\n")
            f.write("| 場景 | 目標類別 | 含GT的Frame數 | TP | FN | FP | FN率 |\n")
            f.write("|------|----------|--------------|----|----|----|----- |\n")

            for scene, target_cls in [
                ("Day_Litter",   "litter"),
                ("Night_Litter", "litter"),
                ("Day_Garbage",  "garbage"),
                ("Night_Garbage","garbage"),
            ]:
                s = all_data[split].get(scene, {}).get(target_cls,
                    {"frames_with_gt": 0, "TP": 0, "FN": 0, "FP": 0})
                fn_rate = (s["FN"] / (s["TP"] + s["FN"])
                           if (s["TP"] + s["FN"]) > 0 else 0.0)
                f.write(f"| {scene} | {target_cls} | {s['frames_with_gt']} "
                        f"| {s['TP']} | {s['FN']} | {s['FP']} | {fn_rate:.2%} |\n")
            f.write("\n")

    print(f"  寫入 {out_path}")


# ── Mode 3: score-dist ────────────────────────────────────────────────────────

def compute_iou(box_a, box_b) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ix1 = max(ax, bx); iy1 = max(ay, by)
    ix2 = min(ax+aw, bx+bw); iy2 = min(ay+ah, by+bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def mode_score_dist(args):
    print("=== score-dist：litter/garbage confidence score 分布 ===")
    from mmdet.apis import init_detector, inference_detector

    ann_file = os.path.join(args.data_root, "annotations/test.json")
    img_dir  = os.path.join(args.data_root, "test")

    with open(ann_file, encoding="utf-8") as f:
        ann_json = json.load(f)

    sorted_cats = sorted(ann_json["categories"], key=lambda c: c["id"])
    cat_map = {c["id"]: idx for idx, c in enumerate(sorted_cats)}
    class_names = [c["name"] for c in sorted_cats]

    gt_data = {}
    for img in ann_json["images"]:
        gt_data[img["id"]] = {"file_name": img["file_name"], "bboxes": [], "classes": []}
    for ann in ann_json["annotations"]:
        iid = ann["image_id"]
        if iid in gt_data:
            gt_data[iid]["bboxes"].append(ann["bbox"])
            gt_data[iid]["classes"].append(cat_map[ann["category_id"]])

    print(f"  載入模型（threshold={SCORE_DIST_THR}）...")
    model = init_detector(args.config, args.checkpoint, device="cuda:0")

    # {cls: {bin_key: {preds, TP, FP}}}
    dist = {cls: {b: {"preds": 0, "TP": 0, "FP": 0}
                  for b in SCORE_BINS}
            for cls in ["litter", "garbage"]}

    total = len(gt_data)
    for i, (img_id, gt_info) in enumerate(gt_data.items()):
        if i % 100 == 0:
            print(f"  進度 {i}/{total}", flush=True)

        img_path = os.path.join(img_dir, gt_info["file_name"])
        det  = inference_detector(model, img_path)
        pred = det.pred_instances
        mask = pred.scores >= SCORE_DIST_THR

        bboxes_xyxy = pred.bboxes[mask].cpu().numpy()
        scores      = pred.scores[mask].cpu().numpy()
        labels      = pred.labels[mask].cpu().numpy()

        for (x1, y1, x2, y2), score, label in zip(bboxes_xyxy, scores, labels):
            cls_name = class_names[label]
            if cls_name not in ("litter", "garbage"):
                continue

            bbox_xywh = [float(x1), float(y1), float(x2-x1), float(y2-y1)]
            score_f   = float(score)

            bin_key = SCORE_BINS[-1]
            for b in SCORE_BINS:
                if b[0] <= score_f < b[1]:
                    bin_key = b
                    break

            dist[cls_name][bin_key]["preds"] += 1

            # 是否 TP：與同類別 GT 的 IoU >= 0.5
            is_tp = any(
                cls_name == class_names[gt_c] and compute_iou(bbox_xywh, gt_b) >= IOU_THR
                for gt_b, gt_c in zip(gt_info["bboxes"], gt_info["classes"])
            )
            if is_tp:
                dist[cls_name][bin_key]["TP"] += 1
            else:
                dist[cls_name][bin_key]["FP"] += 1

    os.makedirs(args.output_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(args.output_dir, "score_distribution.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["score_low", "score_high",
                    "litter_preds", "litter_TP", "litter_FP",
                    "garbage_preds", "garbage_TP", "garbage_FP"])
        for b in SCORE_BINS:
            ld = dist["litter"][b]; gd = dist["garbage"][b]
            w.writerow([b[0], b[1],
                        ld["preds"], ld["TP"], ld["FP"],
                        gd["preds"], gd["TP"], gd["FP"]])
    print(f"  寫入 {csv_path}")

    # Markdown
    md_path = os.path.join(args.output_dir, "score_distribution.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Litter & Garbage Confidence Score 分布（Test Set）\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write(f"- Score 閾值（顯示用）：{SCORE_DIST_THR}（捕捉所有預測框）\n")
        f.write(f"- TP 判定：IoU ≥ {IOU_THR}，同類別 GT\n\n")

        f.write("| Score 區間 | litter 預測框 | 其中TP | 其中FP | garbage 預測框 | 其中TP | 其中FP |\n")
        f.write("|-----------|-------------|--------|--------|--------------|--------|--------|\n")
        for b in SCORE_BINS:
            ld = dist["litter"][b]; gd = dist["garbage"][b]
            f.write(f"| {b[0]:.1f}–{b[1]:.1f} "
                    f"| {ld['preds']} | {ld['TP']} | {ld['FP']} "
                    f"| {gd['preds']} | {gd['TP']} | {gd['FP']} |\n")

        f.write("\n## 分析\n\n")
        for cls in ["litter", "garbage"]:
            total_tp = sum(dist[cls][b]["TP"] for b in SCORE_BINS)
            total_fp = sum(dist[cls][b]["FP"] for b in SCORE_BINS)
            total_all = total_tp + total_fp
            f.write(f"### {cls}\n\n")
            f.write(f"- 總預測框：{total_all}（TP={total_tp}, FP={total_fp}）\n")
            if total_all > 0:
                precision_05 = sum(dist[cls][b]["TP"] for b in SCORE_BINS if b[0] >= 0.5) / \
                               max(1, sum(dist[cls][b]["preds"] for b in SCORE_BINS if b[0] >= 0.5))
                f.write(f"- Score ≥ 0.5 區間 Precision：{precision_05:.2%}\n")
            high_tp_bins = [b for b in SCORE_BINS if dist[cls][b]["TP"] > 0]
            if high_tp_bins:
                best = max(high_tp_bins, key=lambda b: dist[cls][b]["TP"] / max(1, dist[cls][b]["preds"]))
                f.write(f"- Precision 最高區間：{best[0]:.1f}–{best[1]:.1f}\n")
            f.write("\n")

        f.write("### 建議\n\n")
        for cls in ["litter", "garbage"]:
            total_05 = sum(dist[cls][b]["preds"] for b in SCORE_BINS if b[0] >= 0.5)
            total_all = sum(dist[cls][b]["preds"] for b in SCORE_BINS)
            if total_all > 0 and total_05 / total_all < 0.5:
                f.write(f"- {cls}：超過一半的預測框 score < 0.5，"
                        "考慮降低 score threshold 至 0.3 以提升 Recall。\n")
            else:
                f.write(f"- {cls}：score ≥ 0.5 的預測框佔多數，目前 threshold 設定合理。\n")

    print(f"  寫入 {md_path}")


# ── Mode 4: comparison ────────────────────────────────────────────────────────

def _read_final_epoch_metrics(log_path: str, classes: list) -> dict:
    """讀取最後一個 epoch 的指標。"""
    epochs = parse_training_log(log_path, classes)
    if not epochs:
        return {}
    return epochs[-1]


def _read_error_csv_totals(error_dir: str, split: str, classes: list) -> dict:
    """從 per_frame_class_errors.csv 彙總 TP/FN/FP per class。"""
    csv_path = os.path.join(error_dir, f"{split}_per_frame_class_errors.csv")
    if not os.path.exists(csv_path):
        return {}
    totals = {cls: {"TP": 0, "FN": 0, "FP": 0} for cls in classes}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cls = row["class"]
            if cls in totals:
                totals[cls]["TP"] += int(row["TP"])
                totals[cls]["FN"] += int(row["FN"])
                totals[cls]["FP"] += int(row["FP"])
    return totals


def _best_mAP(log_path: str, classes: list) -> tuple:
    epochs = parse_training_log(log_path, classes)
    if not epochs:
        return (None, None)
    best = max(epochs, key=lambda e: e["mAP"] or 0)
    return (best["mAP"], best["epoch"])


def mode_comparison(args):
    print("=== comparison：v8 vs v9 對比 ===")

    v8_epochs = parse_training_log(args.v8_log, V8_CLASSES)
    v9_epochs = parse_training_log(args.v9_log, V9_CLASSES)

    v8_final = v8_epochs[-1] if v8_epochs else {}
    v9_final = v9_epochs[-1] if v9_epochs else {}
    v8_best_mAP, v8_best_ep = _best_mAP(args.v8_log, V8_CLASSES)
    v9_best_mAP, v9_best_ep = _best_mAP(args.v9_log, V9_CLASSES)

    out_path = os.path.join(args.output_dir, "v9_vs_v8_comparison.md")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# v8 vs v9 對比報告\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write("**注意**：v8 的 litter 包含所有垃圾（大型 + 小型）；"
                "v9 的 litter 只含小型垃圾，garbage 為新增的大型垃圾類別。"
                "直接比較 litter AP 時需考慮此差異。\n\n")
        f.write(f"- v8：最終 epoch {v8_final.get('epoch','?')}，"
                f"best mAP={v8_best_mAP:.4f}（Ep{v8_best_ep}）\n")
        f.write(f"- v9：最終 epoch {v9_final.get('epoch','?')}，"
                f"best mAP={v9_best_mAP:.4f}（Ep{v9_best_ep}）\n\n"
                if v9_best_mAP else
                f"- v9：訓練尚未完成或 log 尚無資料\n\n")

        # mAP 對比表
        f.write("## Validation mAP 對比（最終 epoch）\n\n")
        f.write("| 指標 | v8 (final) | v9 (final) |\n")
        f.write("|------|-----------|----------|\n")
        for key, label in [
            ("mAP",         "整體 mAP"),
            ("car_ap",      "car AP"),
            ("litter_ap",   "litter AP *"),
            ("motorcycle_ap","motorcycle AP"),
            ("person_ap",   "person AP"),
        ]:
            v8v = f"{v8_final.get(key) or 0:.4f}" if v8_final.get(key) is not None else "—"
            v9v = f"{v9_final.get(key) or 0:.4f}" if v9_final.get(key) is not None else "—"
            f.write(f"| {label} | {v8v} | {v9v} |\n")
        v9g = v9_final.get("garbage_ap")
        f.write(f"| garbage AP (v9 新增) | — | {v9g:.4f} |\n\n" if v9g is not None else
                "| garbage AP (v9 新增) | — | — |\n\n")
        f.write("\\* v8 litter = 大型+小型垃圾；v9 litter = 小型垃圾（菸蒂/瓶子等）\n\n")

        # 各 split 的 TP/FN 對比
        for split in SPLITS:
            v8_totals = _read_error_csv_totals(args.v8_error_dir, split, V8_CLASSES)
            v9_totals = _read_error_csv_totals(args.v9_error_dir, split, V9_CLASSES)

            if not v8_totals and not v9_totals:
                continue

            f.write(f"---\n\n## {split.upper()} — TP / FN 對比\n\n")
            f.write("| 指標 | v8 | v9 |\n")
            f.write("|------|----|----|\n")

            for cls in ["car", "litter", "motorcycle", "person"]:
                for stat in ["TP", "FN"]:
                    v8v = v8_totals.get(cls, {}).get(stat, "—")
                    v9v = v9_totals.get(cls, {}).get(stat, "—")
                    f.write(f"| {cls} {stat} | {v8v} | {v9v} |\n")
            for stat in ["TP", "FN"]:
                v9v = v9_totals.get("garbage", {}).get(stat, "—")
                f.write(f"| garbage {stat} (v9 新增) | — | {v9v} |\n")

            if v9_totals:
                litter_fn_rate = 0.0
                lt = v9_totals.get("litter", {})
                if lt.get("TP", 0) + lt.get("FN", 0) > 0:
                    litter_fn_rate = lt["FN"] / (lt["TP"] + lt["FN"])
                garbage_fn_rate = 0.0
                gt_ = v9_totals.get("garbage", {})
                if gt_.get("TP", 0) + gt_.get("FN", 0) > 0:
                    garbage_fn_rate = gt_["FN"] / (gt_["TP"] + gt_["FN"])
                f.write(f"| litter FN率 (v9) | — | {litter_fn_rate:.2%} |\n")
                f.write(f"| garbage FN率 (v9) | — | {garbage_fn_rate:.2%} |\n")
            f.write("\n")

    print(f"  寫入 {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Co-DETR v9 post-training analysis")
    p.add_argument("--mode", required=True,
                   choices=["ap-curve", "by-scene", "score-dist", "comparison"])
    p.add_argument("--output-dir", required=True)

    # ap-curve / comparison
    p.add_argument("--v9-log",       help="v9 training_log.md 路徑")
    p.add_argument("--v9-log-extra", help="v9 第一段訓練的 MMEngine log（補 ep1-13）")
    p.add_argument("--v8-log",       help="v8 training_log.md 路徑（comparison 用）")

    # by-scene / comparison
    p.add_argument("--error-dir",    help="v9 error_analysis 輸出目錄")
    p.add_argument("--v8-error-dir", help="v8 error_analysis 輸出目錄（comparison 用）")
    p.add_argument("--v9-error-dir", help="v9 error_analysis 輸出目錄（comparison 用）")

    # score-dist
    p.add_argument("--config",     help="MMDetection config 路徑")
    p.add_argument("--checkpoint", help="模型權重 .pth 路徑")
    p.add_argument("--data-root",  help="資料集根目錄")

    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "ap-curve":
        assert args.v9_log, "--v9-log is required"
        mode_ap_curve(args)

    elif args.mode == "by-scene":
        assert args.error_dir, "--error-dir is required"
        mode_by_scene(args)

    elif args.mode == "score-dist":
        assert args.config and args.checkpoint and args.data_root, \
            "--config, --checkpoint, --data-root are required"
        mode_score_dist(args)

    elif args.mode == "comparison":
        assert args.v8_log and args.v9_log, "--v8-log and --v9-log are required"
        assert args.v8_error_dir and args.v9_error_dir, \
            "--v8-error-dir and --v9-error-dir are required"
        mode_comparison(args)


if __name__ == "__main__":
    main()
