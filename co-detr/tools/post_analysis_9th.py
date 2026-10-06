#!/usr/bin/env python3
"""
Co-DETR 第九次實驗訓練後分析腳本

四個模式，訓練結束後依序執行：

  1. ap-curve    — 從 training_log.md 產出 litter/garbage AP 曲線報告
  2. by-scene    — 從 error_analysis CSV 產出場景分組分析
  3. score-dist  — 對 test set 跑低閾值推論，產出 score 分布報告
  4. comparison  — 對比第八次（chiayi_v10_36e_65vids）vs 第九次（chiayi_v11_36e_65vids）整體結果

執行範例（容器內）：

  # Part 2: AP 曲線
  python /workspace/project/post_analysis_9th.py --mode ap-curve \\
    --log /workspace/project/results/chiayi_v11_36e_65vids/training_log.md \\
    --output-dir /workspace/project/results/chiayi_v11_36e_65vids/error_analysis_score05

  # Part 3: 場景分組（score_thr=0.5）
  python /workspace/project/post_analysis_9th.py --mode by-scene \\
    --error-dir /workspace/project/results/chiayi_v11_36e_65vids/error_analysis_score05 \\
    --output-dir /workspace/project/results/chiayi_v11_36e_65vids/error_analysis_score05 \\
    --output-name litter_garbage_by_scene_score05.md

  # Part 3: 場景分組（score_thr=0.3）
  python /workspace/project/post_analysis_9th.py --mode by-scene \\
    --error-dir /workspace/project/results/chiayi_v11_36e_65vids/error_analysis_score03 \\
    --output-dir /workspace/project/results/chiayi_v11_36e_65vids/error_analysis_score03 \\
    --output-name litter_garbage_by_scene_score03.md

  # Part 4: Score 分布
  python /workspace/project/post_analysis_9th.py --mode score-dist \\
    --config     /workspace/project/configs/co_dino_chiayi_v11_36e_65vids.py \\
    --checkpoint /workspace/project/results/chiayi_v11_36e_65vids/best_coco_litter_precision_epoch_5.pth \\
    --data-root  /workspace/datasets/coco/chiayi_v11_garbage_litter_corrected \\
    --output-dir /workspace/project/results/chiayi_v11_36e_65vids/error_analysis_score05

  # Part 5: 第八次 vs 第九次對比
  python /workspace/project/post_analysis_9th.py --mode comparison \\
    --v7-log      /workspace/hdd2_storage/chiayi_v10_36e_65vids/training_log.md \\
    --v8-log      /workspace/project/results/chiayi_v11_36e_65vids/training_log.md \\
    --v7-error-dir /workspace/hdd2_storage/chiayi_v10_36e_65vids/error_analysis_score05 \\
    --v8-error-dir /workspace/project/results/chiayi_v11_36e_65vids/error_analysis_score05 \\
    --output-dir  /workspace/project/results/chiayi_v11_36e_65vids/error_analysis_score05
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

CLASSES    = ["car", "litter", "motorcycle", "person", "garbage"]
SPLITS     = ["train", "valid", "test"]
SCORE_BINS = [(round(i * 0.1, 1), round((i + 1) * 0.1, 1)) for i in range(10)]
IOU_THR      = 0.5
SCORE_DIST_THR = 0.01


# ── training_log.md 解析 ──────────────────────────────────────────────────────

def parse_training_log(log_path: str) -> list:
    """解析 TrainingMarkdownLogger 產出的 training_log.md。
    回傳 list of dict，每個 epoch 一筆：
    {"epoch", "lr", "mAP", "car_ap", "litter_ap", "motorcycle_ap", "person_ap", "garbage_ap"}
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

        row = {"epoch": epoch_num, "lr": lr, "mAP": metrics.get("bbox_mAP")}
        for cls in CLASSES:
            row[f"{cls}_ap"] = metrics.get(f"{cls}_precision")

        epochs.append(row)

    return epochs


# ── Mode 1: ap-curve ──────────────────────────────────────────────────────────

def mode_ap_curve(args):
    print("=== ap-curve：litter/garbage AP 曲線 ===")
    epochs = parse_training_log(args.log)

    if not epochs:
        print("ERROR: 無法解析 training_log.md，請確認路徑正確。")
        return

    decay_epochs = []
    for i in range(1, len(epochs)):
        if epochs[i]["lr"] and epochs[i - 1]["lr"]:
            if epochs[i]["lr"] < epochs[i - 1]["lr"] * 0.5:
                decay_epochs.append(epochs[i]["epoch"])

    out_path = os.path.join(args.output_dir, "litter_garbage_ap_curve.md")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Litter & Garbage AP 曲線分析（{args.exp_label}實驗）\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write(f"- Config: {args.config_name}\n")
        f.write("- 策略：base_lr=1e-4、backbone_mult=0.01、class_weight=[1,3,1,1,3]、ClassBalancedDataset\n")
        f.write(f"- LR decay epochs: {decay_epochs}\n\n")

        f.write("## Per-Epoch 指標\n\n")
        f.write("| Epoch | LR | mAP | litter AP | garbage AP | car AP | motorcycle AP | person AP |\n")
        f.write("|-------|-----|-----|-----------|------------|--------|---------------|-----------|\n")
        for e in epochs:
            lr_str  = f"{e['lr']:.2e}" if e["lr"] else "—"
            mAP     = f"{e['mAP']:.4f}" if e["mAP"] is not None else "—"
            l_ap    = f"{e.get('litter_ap') or 0:.4f}"
            g_ap    = f"{e.get('garbage_ap') or 0:.4f}"
            car_ap  = f"{e.get('car_ap') or 0:.4f}"
            moto_ap = f"{e.get('motorcycle_ap') or 0:.4f}"
            pers_ap = f"{e.get('person_ap') or 0:.4f}"
            decay_mark = " ⬇LR" if e["epoch"] in decay_epochs else ""
            f.write(f"| {e['epoch']}{decay_mark} | {lr_str} | {mAP} | {l_ap} | {g_ap} "
                    f"| {car_ap} | {moto_ap} | {pers_ap} |\n")

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
        f.write("### Best / Final AP\n\n")
        f.write(f"| 類別 | Best AP | Best Epoch | Final AP (Ep {epochs[-1]['epoch'] if epochs else '?'}) |\n")
        f.write("|------|---------|------------|----------------------------|\n")
        f.write(f"| litter  | {best_l[1]:.4f} | {best_l[0]} | {final_l[1]:.4f} |\n")
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
            f.write("- mAP 在最後 5 epoch 仍有波動。\n")

    print(f"  寫入 {out_path}")


# ── Mode 2: by-scene ──────────────────────────────────────────────────────────

def mode_by_scene(args):
    print("=== by-scene：場景分組分析 ===")

    SCENE_PREFIXES = {
        "Day_Litter":    "litter",
        "Night_Litter":  "litter",
        "Day_Garbage":   "garbage",
        "Night_Garbage": "garbage",
    }

    def get_scene(video_name: str):
        for prefix, cls in SCENE_PREFIXES.items():
            if video_name.startswith(prefix):
                return prefix, cls
        return "Other", None

    all_data = {}

    for split in SPLITS:
        csv_path = os.path.join(args.error_dir, f"{split}_per_frame_class_errors.csv")
        if not os.path.exists(csv_path):
            print(f"  [SKIP] 找不到 {csv_path}")
            continue

        scene_stats = defaultdict(lambda: defaultdict(
            lambda: {"frames_with_gt": 0, "TP": 0, "FN": 0, "FP": 0}
        ))

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

    output_name = getattr(args, "output_name", None) or "litter_garbage_by_scene.md"
    out_path = os.path.join(args.output_dir, output_name)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# Litter & Garbage 場景分組分析（{args.exp_label}實驗）\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write(f"- Error analysis dir: {args.error_dir}\n")
        f.write("> 各場景統計的是「該場景目標類別」的 TP/FN/FP。\n")
        f.write("> Day_Litter / Night_Litter → 統計 litter；Day_Garbage / Night_Garbage → 統計 garbage。\n\n")

        for split in SPLITS:
            if split not in all_data:
                continue
            f.write(f"---\n\n## {split.upper()}\n\n")
            f.write("| 場景 | 目標類別 | 含GT的Frame數 | TP | FN | FP | FN率 |\n")
            f.write("|------|----------|--------------|----|----|----|----- |\n")

            for scene, target_cls in [
                ("Day_Litter",    "litter"),
                ("Night_Litter",  "litter"),
                ("Day_Garbage",   "garbage"),
                ("Night_Garbage", "garbage"),
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
    cat_map     = {c["id"]: idx for idx, c in enumerate(sorted_cats)}
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

    dist = {cls: {b: {"preds": 0, "TP": 0, "FP": 0} for b in SCORE_BINS}
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

            is_tp = any(
                cls_name == class_names[gt_c] and compute_iou(bbox_xywh, gt_b) >= IOU_THR
                for gt_b, gt_c in zip(gt_info["bboxes"], gt_info["classes"])
            )
            if is_tp:
                dist[cls_name][bin_key]["TP"] += 1
            else:
                dist[cls_name][bin_key]["FP"] += 1

    os.makedirs(args.output_dir, exist_ok=True)

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

    md_path = os.path.join(args.output_dir, "score_distribution.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Litter & Garbage Confidence Score 分布（{args.exp_label}實驗 Test Set）\n\n")
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
            total_tp  = sum(dist[cls][b]["TP"]   for b in SCORE_BINS)
            total_fp  = sum(dist[cls][b]["FP"]   for b in SCORE_BINS)
            total_all = total_tp + total_fp
            f.write(f"### {cls}\n\n")
            f.write(f"- 總預測框：{total_all}（TP={total_tp}, FP={total_fp}）\n")
            if total_all > 0:
                precision_05 = (
                    sum(dist[cls][b]["TP"] for b in SCORE_BINS if b[0] >= 0.5) /
                    max(1, sum(dist[cls][b]["preds"] for b in SCORE_BINS if b[0] >= 0.5))
                )
                f.write(f"- Score ≥ 0.5 區間 Precision：{precision_05:.2%}\n")
            high_tp_bins = [b for b in SCORE_BINS if dist[cls][b]["TP"] > 0]
            if high_tp_bins:
                best = max(high_tp_bins, key=lambda b: dist[cls][b]["TP"] / max(1, dist[cls][b]["preds"]))
                f.write(f"- Precision 最高區間：{best[0]:.1f}–{best[1]:.1f}\n")
            f.write("\n")

        f.write("### 建議\n\n")
        for cls in ["litter", "garbage"]:
            total_05  = sum(dist[cls][b]["preds"] for b in SCORE_BINS if b[0] >= 0.5)
            total_all = sum(dist[cls][b]["preds"] for b in SCORE_BINS)
            if total_all > 0 and total_05 / total_all < 0.5:
                f.write(f"- {cls}：超過一半的預測框 score < 0.5，"
                        "考慮降低 score threshold 至 0.3 以提升 Recall。\n")
            else:
                f.write(f"- {cls}：score ≥ 0.5 的預測框佔多數，目前 threshold 設定合理。\n")

    print(f"  寫入 {md_path}")


# ── Mode 4: comparison ────────────────────────────────────────────────────────

def _read_error_csv_totals(error_dir: str, split: str) -> dict:
    csv_path = os.path.join(error_dir, f"{split}_per_frame_class_errors.csv")
    if not os.path.exists(csv_path):
        return {}
    totals = {cls: {"TP": 0, "FN": 0, "FP": 0} for cls in CLASSES}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cls = row["class"]
            if cls in totals:
                totals[cls]["TP"] += int(row["TP"])
                totals[cls]["FN"] += int(row["FN"])
                totals[cls]["FP"] += int(row["FP"])
    return totals


def _best_mAP(epochs: list) -> tuple:
    if not epochs:
        return (None, None)
    best = max(epochs, key=lambda e: e["mAP"] or 0)
    return (best["mAP"], best["epoch"])


def mode_comparison(args):
    bl = args.baseline_label   # baseline 側（v7 slot）
    el = args.exp_label        # 本次實驗側（v8 slot）
    print(f"=== comparison：{bl} vs {el}對比 ===")

    v7_epochs = parse_training_log(args.v7_log)
    v8_epochs = parse_training_log(args.v8_log)

    v7_final = v7_epochs[-1] if v7_epochs else {}
    v8_final = v8_epochs[-1] if v8_epochs else {}
    v7_best_mAP, v7_best_ep = _best_mAP(v7_epochs)
    v8_best_mAP, v8_best_ep = _best_mAP(v8_epochs)

    out_path = os.path.join(args.output_dir, args.comparison_name)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# {bl} vs {el}實驗對比報告\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")

        # 實驗設定差異：由 --config-diff-md 指定（本腳本跨實驗共用，故不內建任何寫死對照表，
        # 避免沿用前次實驗的錯誤內容）。未提供時輸出中性佔位並印警告。
        f.write("## 實驗設定差異\n\n")
        if args.config_diff_md and os.path.exists(args.config_diff_md):
            with open(args.config_diff_md, encoding="utf-8") as df:
                f.write(df.read().rstrip() + "\n\n")
        else:
            print(f"  [WARN] 未指定 --config-diff-md（或檔案不存在）；"
                  f"設定差異段以佔位輸出，請補上以產生正確對照表。")
            f.write(f"> ⚠️ 未提供 `--config-diff-md`，故無法列出 {bl} 與 {el} 的實驗設定差異。\n")
            f.write(f"> 請以 `--config-diff-md <對照檔.md>` 提供兩者設定後重跑本段。\n\n")

        f.write("## 訓練摘要\n\n")
        v7_best_str = f"{v7_best_mAP:.4f}（Ep{v7_best_ep}）" if v7_best_mAP else "—"
        v8_best_str = f"{v8_best_mAP:.4f}（Ep{v8_best_ep}）" if v8_best_mAP else "—"
        f.write(f"- {bl}：共 {v7_final.get('epoch','?')} epoch，best mAP={v7_best_str}\n")
        f.write(f"- {el}：共 {v8_final.get('epoch','?')} epoch，best mAP={v8_best_str}\n\n")

        f.write("## Validation mAP 對比（最終 epoch）\n\n")
        f.write(f"| 指標 | {bl} (final) | {el} (final) | 變化 |\n")
        f.write("|------|--------------|--------------|------|\n")
        for key, label in [
            ("mAP",           "整體 mAP"),
            ("car_ap",        "car AP"),
            ("litter_ap",     "litter AP"),
            ("motorcycle_ap", "motorcycle AP"),
            ("person_ap",     "person AP"),
            ("garbage_ap",    "garbage AP"),
        ]:
            v7v_raw = v7_final.get(key)
            v8v_raw = v8_final.get(key)
            v7v = f"{v7v_raw:.4f}" if v7v_raw is not None else "—"
            v8v = f"{v8v_raw:.4f}" if v8v_raw is not None else "—"
            if v7v_raw is not None and v8v_raw is not None:
                delta = v8v_raw - v7v_raw
                delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            else:
                delta_str = "—"
            f.write(f"| {label} | {v7v} | {v8v} | {delta_str} |\n")
        f.write("\n")

        for split in SPLITS:
            v7_totals = _read_error_csv_totals(args.v7_error_dir, split)
            v8_totals = _read_error_csv_totals(args.v8_error_dir, split)

            if not v7_totals and not v8_totals:
                continue

            f.write(f"---\n\n## {split.upper()} — TP / FN 對比\n\n")
            f.write(f"| 指標 | {bl} | {el} | 變化 |\n")
            f.write("|------|--------|--------|------|\n")

            for cls in CLASSES:
                for stat in ["TP", "FN"]:
                    v7v = v7_totals.get(cls, {}).get(stat, None)
                    v8v = v8_totals.get(cls, {}).get(stat, None)
                    if v7v is not None and v8v is not None:
                        delta = v8v - v7v
                        delta_str = f"+{delta}" if delta >= 0 else str(delta)
                    else:
                        delta_str = "—"
                    v7s = str(v7v) if v7v is not None else "—"
                    v8s = str(v8v) if v8v is not None else "—"
                    f.write(f"| {cls} {stat} | {v7s} | {v8s} | {delta_str} |\n")

            if v8_totals:
                for cls in ["litter", "garbage"]:
                    ct = v8_totals.get(cls, {})
                    fn_rate = (ct["FN"] / (ct["TP"] + ct["FN"])
                               if (ct.get("TP", 0) + ct.get("FN", 0)) > 0 else 0.0)
                    f.write(f"| {cls} FN率 ({el}) | — | {fn_rate:.2%} | — |\n")
            f.write("\n")

        f.write("## 結論\n\n")
        if v7_best_mAP and v8_best_mAP:
            delta_best = v8_best_mAP - v7_best_mAP
            direction = "提升" if delta_best > 0 else "下降"
            f.write(f"- Best mAP：{el} {direction} {abs(delta_best):.4f}（{v7_best_mAP:.4f} → {v8_best_mAP:.4f}）\n")

        v7_l = v7_final.get("litter_ap") or 0
        v8_l = v8_final.get("litter_ap") or 0
        v7_g = v7_final.get("garbage_ap") or 0
        v8_g = v8_final.get("garbage_ap") or 0
        if v8_l > v7_l:
            f.write(f"- litter AP：{el}提升 +{v8_l-v7_l:.4f}（{v7_l:.4f} → {v8_l:.4f}）\n")
        else:
            f.write(f"- litter AP：{el}下降 {v8_l-v7_l:.4f}（{v7_l:.4f} → {v8_l:.4f}）\n")
        if v8_g > v7_g:
            f.write(f"- garbage AP：{el}提升 +{v8_g-v7_g:.4f}（{v7_g:.4f} → {v8_g:.4f}）\n")
        else:
            f.write(f"- garbage AP：{el}下降 {v8_g-v7_g:.4f}（{v7_g:.4f} → {v8_g:.4f}）\n")

    print(f"  寫入 {out_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Co-DETR 第九次實驗 post-training analysis")
    p.add_argument("--mode", required=True,
                   choices=["ap-curve", "by-scene", "score-dist", "comparison"])
    p.add_argument("--output-dir", required=True)

    # 標題/標籤（沿用此腳本時傳入，避免報告寫死成前一次實驗名稱）
    p.add_argument("--exp-label", default="第九次",
                   help="本次實驗標籤（ap-curve/by-scene/score-dist 標題 + comparison 的『本次』側）")
    p.add_argument("--baseline-label", default="第八次",
                   help="comparison 的 baseline 側標籤")
    p.add_argument("--config-name", default="co_dino_chiayi_v11_36e_65vids",
                   help="ap-curve 報告顯示用的 config 名稱")
    p.add_argument("--comparison-name", default="v11_8th_vs_9th_comparison.md",
                   help="comparison 輸出檔名")
    p.add_argument("--config-diff-md",
                   help="comparison『實驗設定差異』表的 markdown 檔（不給則用內建 v10-vs-v11 預設）")

    # ap-curve
    p.add_argument("--log", help="training_log.md 路徑（9th 實驗）")

    # by-scene
    p.add_argument("--error-dir",   help="error_analysis 輸出目錄")
    p.add_argument("--output-name", help="by-scene 輸出檔名（預設 litter_garbage_by_scene.md）")

    # comparison
    p.add_argument("--v7-log",       help="第八次 training_log.md 路徑")
    p.add_argument("--v8-log",       help="第九次 training_log.md 路徑")
    p.add_argument("--v7-error-dir", help="第八次 error_analysis 目錄（score05 版）")
    p.add_argument("--v8-error-dir", help="第九次 error_analysis 目錄（score05 版）")

    # score-dist
    p.add_argument("--config",     help="MMDetection config 路徑")
    p.add_argument("--checkpoint", help="模型權重 .pth 路徑")
    p.add_argument("--data-root",  help="資料集根目錄")

    return p.parse_args()


def main():
    args = parse_args()

    if args.mode == "ap-curve":
        assert args.log, "--log is required"
        mode_ap_curve(args)

    elif args.mode == "by-scene":
        assert args.error_dir, "--error-dir is required"
        mode_by_scene(args)

    elif args.mode == "score-dist":
        assert args.config and args.checkpoint and args.data_root, \
            "--config, --checkpoint, --data-root are required"
        mode_score_dist(args)

    elif args.mode == "comparison":
        assert args.v7_log and args.v8_log, "--v7-log and --v8-log are required"
        assert args.v7_error_dir and args.v8_error_dir, \
            "--v7-error-dir and --v8-error-dir are required"
        mode_comparison(args)


if __name__ == "__main__":
    main()
