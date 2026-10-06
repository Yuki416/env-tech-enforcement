#!/usr/bin/env python3
"""
Co-DETR 錯誤分析腳本 ── 第 17 次 2 類（litter + garbage）專用 fork

★ 為什麼 fork（不共用 tools/error_analysis_v9.py）：
  共用版 CLASSES 硬編 5 類（car/litter/motorcycle/person/garbage），且用「整數索引」比對 GT/pred。
  第 17 次是 2 類模型，pred 輸出的是它自己的 2 類索引（0=litter, 1=garbage），丟進 5 類表會被
  對成 car/litter → 全類別錯位、litter/garbage 的 TP/FN/FP 與 AP 全錯。依「腳本隔離」慣例另 fork。

與共用 v9 版的差異（只有這兩處，其餘完全相同）：
  1. CLASSES = ["litter", "garbage"]（與 17 次 config metainfo 同序 → pred 索引 0/1 直接對齊）。
  2. load_coco_gt：GT 依「類別名稱」映射到 CLASSES 索引，並【丟棄非 litter/garbage 的 GT 框】
     （car/person/motorcycle 不列入）。這樣 GT 與 2 類 pred 落在同一索引空間。
     註：litter/garbage 的逐類 AP@0.5 不受此濾除影響（car GT 本就不進 litter 的 TP/FN）；
        差別僅在「litter pred 落在 car 上」在 5 類版算 class-error、此處算 FP —— 對 litter per-class AP 無差。

⚠️ 與第 13/16 次（5 類原生）對比時，litter/garbage 的 per-class AP 仍可直接比（同為各自類別的 AP）。

在容器內執行（見同夾 post_analysis_17th_v17_2cls.sh，僅跑核心 Part1）：
  python /workspace/project/experiments/17th_v17_own_dynamic_2cls/error_analysis_17th_2cls.py \
    --config     /workspace/project/configs/co_dino_chiayi_v17_own_dynamic_2cls_litter_garbage_36e.py \
    --checkpoint <RESULT_DIR>/best_coco_litter_precision_epoch_24.pth \
    --data-root  /workspace/datasets/coco/chiayi_v17_own_dynamic_litter_head10_manual \
    --output-dir <RESULT_DIR>/error_analysis_score05 \
    --splits train valid test --score-threshold 0.5 \
    --split-ann test=annotations/test_static594.json
"""

import os
import sys
import json
import argparse
import csv
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from PIL import Image

# ── 常數 ─────────────────────────────────────────────────────────────────────

CLASSES = ["litter", "garbage"]  # ★ 2 類，與 17 次 config metainfo 同序（pred 索引 0/1 直接對齊）
LITTER_GARBAGE = {"litter", "garbage"}
IOU_THR = 0.5
SCORE_THR = 0.5

SPLIT_ANN = {
    "train": "annotations/train.json",
    "valid": "annotations/valid.json",
    "test":  "annotations/test.json",
}
SPLIT_IMG_DIR = {"train": "train", "valid": "valid", "test": "test"}


# ── 工具函式 ──────────────────────────────────────────────────────────────────

def extract_video_name(file_name: str) -> str:
    base = os.path.splitext(file_name)[0]
    parts = base.rsplit("_frame_", 1)
    return parts[0] if len(parts) == 2 else base


def compute_iou(box_a, box_b) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def match_detections(gt_boxes, gt_cls, pred_boxes, pred_cls):
    n_gt, n_pred = len(gt_boxes), len(pred_boxes)
    if n_gt == 0:
        return [], [], list(range(n_pred)), []
    if n_pred == 0:
        return [], list(range(n_gt)), [], []

    iou = np.zeros((n_gt, n_pred))
    for i in range(n_gt):
        for j in range(n_pred):
            iou[i, j] = compute_iou(gt_boxes[i], pred_boxes[j])

    pairs = sorted(
        [(iou[i, j], i, j)
         for i in range(n_gt) for j in range(n_pred)
         if iou[i, j] >= IOU_THR],
        reverse=True,
    )

    matched_gt, matched_pred = set(), set()
    tp, cls_err = [], []
    for _, gi, pi in pairs:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        if gt_cls[gi] == pred_cls[pi]:
            tp.append((gi, pi))
        else:
            cls_err.append((gi, pi, CLASSES[gt_cls[gi]], CLASSES[pred_cls[pi]]))

    fn = [i for i in range(n_gt) if i not in matched_gt]
    fp = [j for j in range(n_pred) if j not in matched_pred]
    return tp, fn, fp, cls_err


# ── 資料載入 ──────────────────────────────────────────────────────────────────

def load_coco_gt(json_path: str) -> dict:
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # ★ 2 類版：依「類別名稱」映射到 CLASSES 索引（litter→0, garbage→1），
    #   非 litter/garbage 的類別（car/person/motorcycle）直接丟棄 → GT 與 2 類 pred 同索引空間。
    name_by_id = {c["id"]: c["name"] for c in data["categories"]}
    name_to_idx = {name: idx for idx, name in enumerate(CLASSES)}

    images = {
        img["id"]: {"file_name": img["file_name"], "bboxes": [], "classes": []}
        for img in data["images"]
    }
    for ann in data["annotations"]:
        iid = ann["image_id"]
        if iid not in images:
            continue
        cls_name = name_by_id.get(ann["category_id"])
        if cls_name not in name_to_idx:   # 丟棄 car/person/motorcycle
            continue
        images[iid]["bboxes"].append(ann["bbox"])
        images[iid]["classes"].append(name_to_idx[cls_name])
    return images


# ── 推論 ──────────────────────────────────────────────────────────────────────

def run_inference(config_file: str, checkpoint_file: str,
                  img_dir: str, images_info: dict,
                  score_thr: float = SCORE_THR) -> dict:
    from mmdet.apis import init_detector, inference_detector

    print(f"  載入模型：{checkpoint_file}")
    model = init_detector(config_file, checkpoint_file, device="cuda:0")

    results = {}
    total = len(images_info)
    for i, (img_id, info) in enumerate(images_info.items()):
        if i % 200 == 0:
            print(f"  推論進度 {i}/{total} ...", flush=True)

        img_path = os.path.join(img_dir, info["file_name"])
        det = inference_detector(model, img_path)
        pred = det.pred_instances
        mask = pred.scores >= score_thr

        bboxes_xyxy = pred.bboxes[mask].cpu().numpy()
        scores      = pred.scores[mask].cpu().numpy()
        labels      = pred.labels[mask].cpu().numpy()

        bboxes_xywh = [
            [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            for x1, y1, x2, y2 in bboxes_xyxy
        ]
        results[img_id] = {
            "bboxes":  bboxes_xywh,
            "classes": labels.tolist(),
            "scores":  scores.tolist(),
        }
    return results


# ── 分析 ──────────────────────────────────────────────────────────────────────

def analyze_split(gt_data: dict, pred_data: dict) -> list:
    results = []
    for img_id, gt_info in gt_data.items():
        pred_info = pred_data.get(img_id, {"bboxes": [], "classes": [], "scores": []})
        tp, fn, fp, cls_err = match_detections(
            gt_info["bboxes"], gt_info["classes"],
            pred_info["bboxes"], pred_info["classes"],
        )
        results.append({
            "img_id":       img_id,
            "file_name":    gt_info["file_name"],
            "video_name":   extract_video_name(gt_info["file_name"]),
            "tp":           tp,
            "fn":           fn,
            "fp":           fp,
            "cls_err":      cls_err,
            "gt_boxes":     gt_info["bboxes"],
            "gt_cls":       gt_info["classes"],
            "pred_boxes":   pred_info["bboxes"],
            "pred_cls":     pred_info["classes"],
            "pred_scores":  pred_info["scores"],
            "total_errors": len(fn) + len(fp) + len(cls_err),
        })
    return results


# ── 指標計算 ──────────────────────────────────────────────────────────────────

def compute_confusion_matrix(results: list):
    n = len(CLASSES)
    cm        = np.zeros((n, n), dtype=int)
    tp_count  = np.zeros(n, dtype=int)
    fn_count  = np.zeros(n, dtype=int)
    fp_count  = np.zeros(n, dtype=int)

    for r in results:
        for gi, _ in r["tp"]:
            tp_count[r["gt_cls"][gi]] += 1
        for gi in r["fn"]:
            fn_count[r["gt_cls"][gi]] += 1
        for pi in r["fp"]:
            fp_count[r["pred_cls"][pi]] += 1
        for gi, pi, gt_name, pred_name in r["cls_err"]:
            gi_cls = CLASSES.index(gt_name)
            pi_cls = CLASSES.index(pred_name)
            cm[gi_cls][pi_cls] += 1
            fn_count[gi_cls] += 1
            fp_count[pi_cls] += 1

    return cm, tp_count, fn_count, fp_count


def compute_per_class_metrics(tp_count, fn_count, fp_count) -> dict:
    metrics = {}
    for i, cls in enumerate(CLASSES):
        tp = int(tp_count[i])
        fn = int(fn_count[i])
        fp = int(fp_count[i])
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)
        metrics[cls] = {
            "TP": tp, "FN": fn, "FP": fp,
            "Precision": precision, "Recall": recall, "F1": f1,
        }
    return metrics


def compute_coco_ap(ann_file: str, pred_data: dict) -> dict:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(ann_file)
    with open(ann_file, encoding="utf-8") as f:
        ann_json = json.load(f)
    # ★ 2 類版：pred 的 category_id 與 eval 的 catId 都必須依「名稱」對回 json 的 category id
    #   （模型 idx 0=litter,1=garbage；json 的 litter=2, garbage=5）。
    #   共用版誤用 sorted(id)[cls_idx] → litter(0)→id1=car、garbage(1)→id2=litter → 拿錯 GT 比 → AP=0。
    name_to_catid = {c["name"]: c["id"] for c in ann_json["categories"]}

    coco_results = []
    for img_id, pred in pred_data.items():
        for bbox, cls_idx, score in zip(pred["bboxes"], pred["classes"], pred["scores"]):
            coco_results.append({
                "image_id":    img_id,
                "category_id": name_to_catid[CLASSES[cls_idx]],
                "bbox":        [round(v, 2) for v in bbox],
                "score":       round(float(score), 4),
            })

    if not coco_results:
        return {cls: {"AP@0.5:0.95": 0.0, "AP@0.5": 0.0} for cls in CLASSES}

    coco_dt = coco_gt.loadRes(coco_results)
    per_class_ap = {}
    for i, cls_name in enumerate(CLASSES):
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.params.catIds = [name_to_catid[cls_name]]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        per_class_ap[cls_name] = {
            "AP@0.5:0.95": float(coco_eval.stats[0]),
            "AP@0.5":      float(coco_eval.stats[1]),
        }
    return per_class_ap


# ── CSV 輸出 ──────────────────────────────────────────────────────────────────

def write_per_frame_csv(results: list, split: str, output_dir: str):
    path = os.path.join(output_dir, f"{split}_per_frame_errors.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "file_name", "video_name",
                    "TP", "FN", "FP", "ClassError", "TotalErrors"])
        for r in sorted(results, key=lambda x: -x["total_errors"]):
            w.writerow([
                split, r["file_name"], r["video_name"],
                len(r["tp"]), len(r["fn"]), len(r["fp"]),
                len(r["cls_err"]), r["total_errors"],
            ])
    print(f"  寫入 {path}")


def write_per_frame_class_csv(results: list, split: str, output_dir: str):
    """Per-frame per-class TP/FN/FP，供 litter_garbage_by_scene 分析用。"""
    path = os.path.join(output_dir, f"{split}_per_frame_class_errors.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "file_name", "video_name", "class", "TP", "FN", "FP"])
        for r in results:
            stats = {cls: {"TP": 0, "FN": 0, "FP": 0} for cls in CLASSES}
            for gi, _ in r["tp"]:
                stats[CLASSES[r["gt_cls"][gi]]]["TP"] += 1
            for gi in r["fn"]:
                stats[CLASSES[r["gt_cls"][gi]]]["FN"] += 1
            for pi in r["fp"]:
                stats[CLASSES[r["pred_cls"][pi]]]["FP"] += 1
            for gi, pi, gt_name, pred_name in r["cls_err"]:
                stats[gt_name]["FN"] += 1
                stats[pred_name]["FP"] += 1
            for cls in CLASSES:
                w.writerow([split, r["file_name"], r["video_name"], cls,
                            stats[cls]["TP"], stats[cls]["FN"], stats[cls]["FP"]])
    print(f"  寫入 {path}")


def write_confusion_matrix_csv(cm, tp_count, fn_count, fp_count,
                                split: str, output_dir: str):
    path = os.path.join(output_dir, f"{split}_confusion_matrix.csv")
    n = len(CLASSES)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["GT \\ Pred"] + CLASSES + ["FN (漏偵測)", "TP"])
        for i, cls in enumerate(CLASSES):
            w.writerow([cls] + [cm[i][j] for j in range(n)] +
                       [fn_count[i], tp_count[i]])
        w.writerow(["FP (誤報)"] + [fp_count[j] for j in range(n)] + ["", ""])
    print(f"  寫入 {path}")


def write_per_video_csv(results: list, split: str, output_dir: str) -> dict:
    stats = defaultdict(lambda: {"TP": 0, "FN": 0, "FP": 0, "ClassError": 0, "Frames": 0})
    for r in results:
        v = r["video_name"]
        stats[v]["TP"]         += len(r["tp"])
        stats[v]["FN"]         += len(r["fn"])
        stats[v]["FP"]         += len(r["fp"])
        stats[v]["ClassError"] += len(r["cls_err"])
        stats[v]["Frames"]     += 1

    path = os.path.join(output_dir, f"{split}_per_video_errors.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_name", "frames", "TP", "FN", "FP", "ClassError", "TotalErrors"])
        for vname, s in sorted(stats.items(),
                                key=lambda x: -(x[1]["FN"] + x[1]["FP"] + x[1]["ClassError"])):
            total = s["FN"] + s["FP"] + s["ClassError"]
            w.writerow([vname, s["Frames"], s["TP"], s["FN"], s["FP"], s["ClassError"], total])
    print(f"  寫入 {path}")
    return dict(stats)


# ── Markdown 報告 ─────────────────────────────────────────────────────────────

def write_markdown_report(split_summaries: dict, output_dir: str,
                          score_thr: float = SCORE_THR, exp_name: str = "chiayi_v9_36e"):
    path = os.path.join(output_dir, "error_analysis_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Error Analysis Report — {exp_name}\n\n")
        f.write(f"Generated: {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write(f"- Classes: {CLASSES}\n")
        f.write(f"- IoU threshold: {IOU_THR}\n")
        f.write(f"- Score threshold: {score_thr}\n\n")

        for split, s in split_summaries.items():
            f.write(f"---\n\n## {split.upper()}\n\n")

            f.write("### Per-Class Metrics\n\n")
            f.write("| Class | AP@0.5 | AP@0.5:0.95 | Precision | Recall | F1 | TP | FP | FN |\n")
            f.write("|-------|--------|-------------|-----------|--------|----|----|----|----|  \n")
            for cls in CLASSES:
                m  = s["per_class_metrics"][cls]
                ap = s["coco_ap"].get(cls, {})
                ap50   = f'{ap.get("AP@0.5",      0):.3f}'
                ap5095 = f'{ap.get("AP@0.5:0.95", 0):.3f}'
                f.write(
                    f"| {cls} | {ap50} | {ap5095} | "
                    f"{m['Precision']:.3f} | {m['Recall']:.3f} | {m['F1']:.3f} | "
                    f"{m['TP']} | {m['FP']} | {m['FN']} |\n"
                )

            cm       = s["confusion_matrix"]
            tp_count = s["tp_count"]
            fn_count = s["fn_count"]
            fp_count = s["fp_count"]
            n = len(CLASSES)

            f.write("\n### Detection Confusion Matrix\n\n")
            f.write("> 列 = GT 類別，欄 = 預測類別\n\n")
            header = "| GT \\ Pred | " + " | ".join(CLASSES) + " | FN (漏偵測) | TP |\n"
            sep    = "|" + "---|" * (n + 3) + "\n"
            f.write(header + sep)
            for i, cls in enumerate(CLASSES):
                row = f"| **{cls}** | " + " | ".join(str(cm[i][j]) for j in range(n))
                row += f" | {fn_count[i]} | {tp_count[i]} |\n"
                f.write(row)
            f.write("| **FP (誤報)** | " +
                    " | ".join(str(fp_count[j]) for j in range(n)) + " | | |\n\n")

            f.write("### Per-Class Error Breakdown\n\n")
            f.write("| Class | TP | FN (漏偵測) | FP (誤報) | FN率 | FP率 |\n")
            f.write("|-------|----|------------|-----------|------|------|\n")
            for cls in CLASSES:
                m = s["per_class_metrics"][cls]
                fn_rate = m["FN"] / (m["TP"] + m["FN"]) if (m["TP"] + m["FN"]) > 0 else 0
                fp_rate = m["FP"] / (m["TP"] + m["FP"]) if (m["TP"] + m["FP"]) > 0 else 0
                f.write(f"| {cls} | {m['TP']} | {m['FN']} | {m['FP']} "
                        f"| {fn_rate:.2%} | {fp_rate:.2%} |\n")

            f.write("\n### Top 20 Videos by Error Count\n\n")
            f.write("| # | Video | Frames | TP | FN | FP | ClassError | TotalErrors |\n")
            f.write("|---|-------|--------|----|----|----|------------|-------------|\n")
            sorted_vids = sorted(
                s["video_stats"].items(),
                key=lambda x: -(x[1]["FN"] + x[1]["FP"] + x[1]["ClassError"]),
            )
            for rank, (vname, vs) in enumerate(sorted_vids[:20], 1):
                total = vs["FN"] + vs["FP"] + vs["ClassError"]
                f.write(f"| {rank} | {vname} | {vs['Frames']} | {vs['TP']} "
                        f"| {vs['FN']} | {vs['FP']} | {vs['ClassError']} | {total} |\n")
            f.write("\n")

    print(f"  寫入 {path}")


# ── 視覺化 ────────────────────────────────────────────────────────────────────

def _dashed_rect(ax, x, y, w, h, color, lw=2):
    rect = patches.Rectangle(
        (x, y), w, h,
        linewidth=lw, edgecolor=color, facecolor="none", linestyle="--",
    )
    ax.add_patch(rect)


def visualize_top_errors(results: list, img_dir: str,
                          split_output_dir: str, top_n: int = 20):
    vis_dir = os.path.join(split_output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    top = sorted(results, key=lambda r: -r["total_errors"])[:top_n]

    legend_elements = [
        Line2D([0], [0], color="#00bb00", lw=2,                label="GT (正確偵測)"),
        Line2D([0], [0], color="#0077ff", lw=2, linestyle="--", label="FN (漏偵測)"),
        Line2D([0], [0], color="red",     lw=2,                label="FP (誤報)"),
        Line2D([0], [0], color="orange",  lw=2, linestyle="--", label="Class Error"),
    ]

    for rank, r in enumerate(top, 1):
        img_path = os.path.join(img_dir, r["file_name"])
        if not os.path.exists(img_path):
            print(f"  [WARN] 找不到圖片：{img_path}")
            continue

        img = np.array(Image.open(img_path).convert("RGB"))
        h_img, w_img = img.shape[:2]

        fig, ax = plt.subplots(1, figsize=(w_img / 120, h_img / 120))
        ax.imshow(img)

        tp_gt_set = {gi for gi, _ in r["tp"]}
        for gi in tp_gt_set:
            x, y, w, h = r["gt_boxes"][gi]
            ax.add_patch(patches.Rectangle(
                (x, y), w, h, linewidth=1.5, edgecolor="#00bb00", facecolor="none"))
            ax.text(x, max(0, y - 4), f"GT:{CLASSES[r['gt_cls'][gi]]}",
                    color="#00bb00", fontsize=6, va="bottom",
                    bbox=dict(facecolor="white", alpha=0.5, pad=1, linewidth=0))

        for gi in r["fn"]:
            x, y, w, h = r["gt_boxes"][gi]
            _dashed_rect(ax, x, y, w, h, "#0077ff")
            ax.text(x, max(0, y - 4), f"FN:{CLASSES[r['gt_cls'][gi]]}",
                    color="#0077ff", fontsize=6, va="bottom",
                    bbox=dict(facecolor="white", alpha=0.5, pad=1, linewidth=0))

        for gi, pi, gt_name, pred_name in r["cls_err"]:
            x, y, w, h = r["gt_boxes"][gi]
            _dashed_rect(ax, x, y, w, h, "orange")
            ax.text(x, max(0, y - 4), f"CE:{gt_name}→{pred_name}",
                    color="orange", fontsize=6, va="bottom",
                    bbox=dict(facecolor="white", alpha=0.5, pad=1, linewidth=0))

        for pi in r["fp"]:
            x, y, w, h = r["pred_boxes"][pi]
            score = r["pred_scores"][pi]
            ax.add_patch(patches.Rectangle(
                (x, y), w, h, linewidth=1.5, edgecolor="red", facecolor="none"))
            ax.text(x, min(h_img, y + h + 4),
                    f"FP:{CLASSES[r['pred_cls'][pi]]}({score:.2f})",
                    color="red", fontsize=6, va="top",
                    bbox=dict(facecolor="white", alpha=0.5, pad=1, linewidth=0))

        n_tp = len(r["tp"]); n_fn = len(r["fn"]); n_fp = len(r["fp"]); n_ce = len(r["cls_err"])
        title = (f"#{rank}  {os.path.basename(r['file_name'])}\n"
                 f"TP={n_tp}  FN={n_fn}(藍虛線)  FP={n_fp}(紅)  ClassErr={n_ce}(橘虛線)")
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        ax.legend(handles=legend_elements, loc="upper right", fontsize=7)

        out_path = os.path.join(vis_dir, f"rank{rank:02d}_{os.path.basename(r['file_name'])}")
        plt.tight_layout()
        plt.savefig(out_path, dpi=80, bbox_inches="tight")
        plt.close()
        print(f"  視覺化 {rank}/{top_n}: {os.path.basename(out_path)}")


def visualize_litter_garbage_fp(results: list, img_dir: str,
                                  output_dir: str, split: str):
    """儲存所有含 litter 或 garbage FP 的 frame。綠框=GT，紅框=FP。"""
    vis_dir = os.path.join(output_dir, "litter_garbage_fp_visualizations", split)
    os.makedirs(vis_dir, exist_ok=True)

    count = 0
    for r in results:
        lg_fp_idx = [pi for pi in r["fp"]
                     if CLASSES[r["pred_cls"][pi]] in LITTER_GARBAGE]
        if not lg_fp_idx:
            continue

        img_path = os.path.join(img_dir, r["file_name"])
        if not os.path.exists(img_path):
            continue

        img = np.array(Image.open(img_path).convert("RGB"))
        h_img, w_img = img.shape[:2]

        fig, ax = plt.subplots(1, figsize=(w_img / 100, h_img / 100))
        ax.imshow(img)

        for bbox, gt_c in zip(r["gt_boxes"], r["gt_cls"]):
            x, y, w, h = bbox
            ax.add_patch(patches.Rectangle(
                (x, y), w, h, linewidth=2, edgecolor="#00bb00", facecolor="none"))
            ax.text(x, max(0, y - 4), CLASSES[gt_c], color="#00bb00", fontsize=11,
                    va="bottom", fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.75, pad=2, linewidth=0))

        for pi in lg_fp_idx:
            x, y, w, h = r["pred_boxes"][pi]
            score = r["pred_scores"][pi]
            cls_name = CLASSES[r["pred_cls"][pi]]
            ax.add_patch(patches.Rectangle(
                (x, y), w, h, linewidth=2, edgecolor="red", facecolor="none"))
            ax.text(x, min(h_img, y + h + 4), f"{cls_name}({score:.2f})",
                    color="red", fontsize=11, va="top", fontweight="bold",
                    bbox=dict(facecolor="white", alpha=0.75, pad=2, linewidth=0))

        ax.set_title(
            f"{os.path.basename(r['file_name'])} | {len(lg_fp_idx)} litter/garbage FP",
            fontsize=11)
        ax.axis("off")

        out_path = os.path.join(vis_dir, os.path.basename(r["file_name"]))
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        count += 1

    print(f"  [{split}] litter/garbage FP 視覺化：{count} 張 → {vis_dir}")
    return count


# ── 主流程 ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Co-DETR error analysis (v9 — 5 classes)")
    p.add_argument("--config",      required=True)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--data-root",   required=True)
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--splits", nargs="+", default=["train", "valid", "test"])
    p.add_argument("--score-threshold", type=float, default=SCORE_THR)
    # 覆寫 split → annotation 檔的對應（不給則沿用 SPLIT_ANN 預設，向後相容）。
    # 用途：對同一個 img 目錄的「子集標註」做分析，例如第 15 次的主判定
    #   --split-ann test=annotations/test_static594.json
    # （v16 的 test 是 655 張含新增 clip，但與第 13 次可比的只有原 v11 的 594 張）
    p.add_argument("--split-ann", nargs="+", default=[], metavar="SPLIT=REL_PATH",
                   help="覆寫某 split 的 annotation 相對路徑，例：test=annotations/test_static594.json")
    # 覆寫 split → 影像子目錄（子集標註通常仍讀原 split 的影像目錄，故預設不需動）
    p.add_argument("--split-img-dir", nargs="+", default=[], metavar="SPLIT=DIRNAME",
                   help="覆寫某 split 的影像目錄名，例：test=test")
    return p.parse_args()


def _apply_overrides(pairs, table, what):
    """把 ['test=annotations/x.json', ...] 套進對應表；key 必須是已知 split。"""
    for kv in pairs:
        if "=" not in kv:
            raise SystemExit(f"--{what} 需為 SPLIT=VALUE 格式，收到: {kv!r}")
        k, v = kv.split("=", 1)
        if k not in table:
            raise SystemExit(f"--{what} 的 split {k!r} 不存在（可用: {sorted(table)}）")
        print(f"  [override] {what}: {k} → {v}")
        table[k] = v


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 每次執行都用副本，避免覆寫汙染模組層級的預設值
    split_ann = dict(SPLIT_ANN)
    split_img_dir = dict(SPLIT_IMG_DIR)
    _apply_overrides(args.split_ann, split_ann, "split-ann")
    _apply_overrides(args.split_img_dir, split_img_dir, "split-img-dir")

    split_summaries = {}

    for split in args.splits:
        print(f"\n{'='*60}")
        print(f"  分析 {split}")
        print(f"{'='*60}")

        ann_file = os.path.join(args.data_root, split_ann[split])
        img_dir  = os.path.join(args.data_root, split_img_dir[split])

        print("  載入 GT annotations ...")
        gt_data = load_coco_gt(ann_file)

        print("  執行推論 ...")
        pred_data = run_inference(args.config, args.checkpoint, img_dir, gt_data,
                                  score_thr=args.score_threshold)

        print("  比對 GT 與預測 ...")
        results = analyze_split(gt_data, pred_data)

        cm, tp_count, fn_count, fp_count = compute_confusion_matrix(results)
        per_class_metrics = compute_per_class_metrics(tp_count, fn_count, fp_count)

        print("  計算 COCO AP ...")
        coco_ap = compute_coco_ap(ann_file, pred_data)

        write_per_frame_csv(results, split, args.output_dir)
        write_per_frame_class_csv(results, split, args.output_dir)
        write_confusion_matrix_csv(cm, tp_count, fn_count, fp_count, split, args.output_dir)
        video_stats = write_per_video_csv(results, split, args.output_dir)

        split_output_dir = os.path.join(args.output_dir, split)
        os.makedirs(split_output_dir, exist_ok=True)
        print(f"  視覺化錯誤最多的前 20 張 ({split}) ...")
        visualize_top_errors(results, img_dir, split_output_dir)

        print(f"  視覺化 litter/garbage FP ({split}) ...")
        visualize_litter_garbage_fp(results, img_dir, args.output_dir, split)

        split_summaries[split] = {
            "confusion_matrix":  cm,
            "tp_count":          tp_count,
            "fn_count":          fn_count,
            "fp_count":          fp_count,
            "per_class_metrics": per_class_metrics,
            "coco_ap":           coco_ap,
            "video_stats":       video_stats,
        }

    # 實驗名稱：由 config 檔名推導（co_dino_chiayi_vXX_...py → chiayi_vXX_...），
    # 避免報告標題寫死成前一次實驗的名字。
    exp_name = os.path.splitext(os.path.basename(args.config))[0]
    if exp_name.startswith("co_dino_"):
        exp_name = exp_name[len("co_dino_"):]

    print("\n  寫入總報告 ...")
    write_markdown_report(split_summaries, args.output_dir,
                          score_thr=args.score_threshold, exp_name=exp_name)
    print(f"\n完成。所有結果存於：{args.output_dir}")


if __name__ == "__main__":
    main()
