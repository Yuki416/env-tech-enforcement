"""
第七次 vs 第八次實驗：litter/garbage TP frame 視覺化比較圖。

輸出結構：
  results/chiayi_v10_36e_65vids/litter_tp_comparison/
    7th/train/  7th/valid/  7th/test/
    8th/train/  8th/valid/  8th/test/

每張圖：
  綠色實線  = GT（被偵測到，TP）
  紅色虛線  = GT（漏偵測，FN）
  藍色實線  = TP 預測框

使用方式（容器內）：
  python /workspace/project/visualize_litter_tp_comparison_8th.py \\
    --v7-checkpoint /workspace/project/results/7th_expt_v9_loss_compensate_36e/best_coco_litter_precision_epoch_5.pth \\
    --v8-checkpoint /workspace/project/results/chiayi_v10_36e_65vids/best_coco_litter_precision_epoch_X.pth \\
    --v7-ea-dir     /workspace/project/results/7th_expt_v9_loss_compensate_36e/error_analysis_score05 \\
    --v8-ea-dir     /workspace/project/results/chiayi_v10_36e_65vids/error_analysis_score05 \\
    --output-dir    /workspace/project/results/chiayi_v10_36e_65vids/litter_tp_comparison
"""

import os
import csv
import json
import argparse
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# ── 常數 ─────────────────────────────────────────────────────────────────────

CLASSES       = ("car", "litter", "motorcycle", "person", "garbage")
TARGET_CLASSES = {"litter", "garbage"}
SCORE_THR     = 0.5
IOU_THR       = 0.5

V7_CONFIG    = "/workspace/hdd2_storage/chiayi_v9_36e/co_dino_chiayi_v9_36e.py"
V8_CONFIG    = "/workspace/hdd2_storage/chiayi_v10_36e_65vids/co_dino_chiayi_v10_36e_65vids.py"
V7_DATA_ROOT = "/workspace/hdd2_storage/coco/chiayi_v9_garbage_litter"
V8_DATA_ROOT = "/workspace/hdd2_storage/coco/chiayi_v10_garbage_litter"

SPLITS = ["train", "valid", "test"]


# ── COCO GT 載入 ──────────────────────────────────────────────────────────────

def load_coco_gt(json_path):
    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)
    cls_list = [c["name"] for c in sorted(d["categories"], key=lambda c: c["id"])]
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(sorted(d["categories"], key=lambda c: c["id"]))}
    gt = {}
    for img in d["images"]:
        fn = img["file_name"]
        base = os.path.splitext(fn)[0]
        video = base.rsplit("_frame_", 1)[0] if "_frame_" in base else base
        gt[img["id"]] = {"file_name": fn, "video_name": video,
                         "bboxes": [], "classes": []}
    for ann in d["annotations"]:
        iid = ann["image_id"]
        if iid in gt:
            gt[iid]["bboxes"].append(ann["bbox"])
            gt[iid]["classes"].append(cat_id_to_idx[ann["category_id"]])
    return gt


# ── IoU & matching ────────────────────────────────────────────────────────────

def iou_xywh(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2),   min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter / union if union > 0 else 0.0


def match(gt_boxes, gt_cls, pred_boxes, pred_cls):
    matched_gt, matched_pred = set(), set()
    tp_pairs = []
    for pi, (pb, pc) in enumerate(zip(pred_boxes, pred_cls)):
        best_iou, best_gi = 0, -1
        for gi, (gb, gc) in enumerate(zip(gt_boxes, gt_cls)):
            if gi in matched_gt or gc != pc:
                continue
            v = iou_xywh(gb, pb)
            if v > best_iou:
                best_iou, best_gi = v, gi
        if best_iou >= IOU_THR:
            matched_gt.add(best_gi)
            matched_pred.add(pi)
            tp_pairs.append((best_gi, pi))
    fn = [gi for gi in range(len(gt_boxes)) if gi not in matched_gt]
    fp = [pi for pi in range(len(pred_boxes)) if pi not in matched_pred]
    return tp_pairs, fn, fp


# ── 推論 ──────────────────────────────────────────────────────────────────────

def run_inference(config, checkpoint, img_dir, gt_data):
    from mmdet.apis import init_detector, inference_detector
    model = init_detector(config, checkpoint, device="cuda:0")
    pred = {}
    files = list(gt_data.keys())
    for i, img_id in enumerate(files):
        if i % 100 == 0:
            print(f"    推論 {i}/{len(files)} ...", flush=True)
        path = os.path.join(img_dir, gt_data[img_id]["file_name"])
        if not os.path.exists(path):
            pred[img_id] = {"bboxes": [], "classes": [], "scores": []}
            continue
        result = inference_detector(model, path)
        bboxes, classes, scores = [], [], []
        if hasattr(result, "pred_instances"):
            inst = result.pred_instances
            for j in range(len(inst.bboxes)):
                sc = float(inst.scores[j])
                if sc < SCORE_THR:
                    continue
                x1, y1, x2, y2 = inst.bboxes[j].tolist()
                bboxes.append([x1, y1, x2 - x1, y2 - y1])
                classes.append(int(inst.labels[j]))
                scores.append(sc)
        pred[img_id] = {"bboxes": bboxes, "classes": classes, "scores": scores}
    return pred


# ── 視覺化 ────────────────────────────────────────────────────────────────────

def draw_frame(img_path, gt_boxes, gt_cls, pred_boxes, pred_cls, pred_scores,
               out_path, title=""):
    img = np.array(Image.open(img_path).convert("RGB"))
    h, w = img.shape[:2]
    fig, ax = plt.subplots(1, figsize=(w / 120, h / 120))
    ax.imshow(img)

    target_idx = {list(CLASSES).index(c) for c in TARGET_CLASSES if c in CLASSES}
    tp_pairs, fn_idx, _ = match(gt_boxes, gt_cls, pred_boxes, pred_cls)
    tp_gt   = {gi for gi, _ in tp_pairs}
    tp_pred = {pi for _, pi in tp_pairs}

    for gi, pi in tp_pairs:
        if gt_cls[gi] not in target_idx:
            continue
        x, y, bw, bh = gt_boxes[gi]
        ax.add_patch(patches.Rectangle((x, y), bw, bh, lw=2,
                                        edgecolor="#00cc00", facecolor="none"))
        ax.text(x, max(0, y - 4), f"GT:{CLASSES[gt_cls[gi]]}",
                color="#00cc00", fontsize=8, fontweight="bold", va="bottom",
                bbox=dict(facecolor="white", alpha=0.6, pad=1, linewidth=0))
        x, y, bw, bh = pred_boxes[pi]
        ax.add_patch(patches.Rectangle((x, y), bw, bh, lw=2,
                                        edgecolor="#2288ff", facecolor="none"))
        ax.text(x, min(h, y + bh + 4),
                f"TP:{CLASSES[pred_cls[pi]]}({pred_scores[pi]:.2f})",
                color="#2288ff", fontsize=8, fontweight="bold", va="top",
                bbox=dict(facecolor="white", alpha=0.6, pad=1, linewidth=0))

    for gi in fn_idx:
        if gt_cls[gi] not in target_idx:
            continue
        x, y, bw, bh = gt_boxes[gi]
        for dx in np.arange(0, bw, 8):
            ax.plot([x+dx, min(x+dx+4, x+bw)], [y, y], color="red", lw=1.5)
            ax.plot([x+dx, min(x+dx+4, x+bw)], [y+bh, y+bh], color="red", lw=1.5)
        for dy in np.arange(0, bh, 8):
            ax.plot([x, x], [y+dy, min(y+dy+4, y+bh)], color="red", lw=1.5)
            ax.plot([x+bw, x+bw], [y+dy, min(y+dy+4, y+bh)], color="red", lw=1.5)
        ax.text(x, max(0, y - 4), f"FN:{CLASSES[gt_cls[gi]]}",
                color="red", fontsize=8, va="bottom",
                bbox=dict(facecolor="white", alpha=0.6, pad=1, linewidth=0))

    legend = [
        Line2D([0], [0], color="#00cc00", lw=2, label="GT (TP)"),
        Line2D([0], [0], color="#2288ff", lw=2, label="TP prediction"),
        Line2D([0], [0], color="red",    lw=2, linestyle="--", label="FN (missed)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()


# ── 主流程 ────────────────────────────────────────────────────────────────────

def process_run(label, config, checkpoint, data_root, ea_dir, output_dir):
    for split in SPLITS:
        csv_path = os.path.join(ea_dir, f"{split}_per_frame_class_errors.csv")
        if not os.path.exists(csv_path):
            print(f"  [{label}/{split}] 找不到 {csv_path}，跳過")
            continue

        # 找出有 litter/garbage TP 的 frame
        tp_frames = set()
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["class"] in TARGET_CLASSES and int(row["TP"]) > 0:
                    tp_frames.add(row["file_name"])

        if not tp_frames:
            print(f"  [{label}/{split}] 無 litter/garbage TP，跳過")
            continue

        print(f"  [{label}/{split}] 有 TP 的 frame：{len(tp_frames)} 張，載入 GT ...")
        ann_path = os.path.join(data_root, f"annotations/{split}.json")
        gt_data  = load_coco_gt(ann_path)

        # 只保留有 TP 的 frame
        tp_gt = {iid: v for iid, v in gt_data.items()
                 if v["file_name"] in tp_frames}

        if not tp_gt:
            print(f"  [{label}/{split}] GT 過濾後無資料，跳過")
            continue

        print(f"  [{label}/{split}] 執行推論（{len(tp_gt)} 張）...")
        img_dir = os.path.join(data_root, split)
        pred = run_inference(config, checkpoint, img_dir, tp_gt)

        out_dir = os.path.join(output_dir, label, split)
        os.makedirs(out_dir, exist_ok=True)

        print(f"  [{label}/{split}] 產生視覺化 ...")
        for iid, ginfo in tp_gt.items():
            pinfo    = pred.get(iid, {"bboxes": [], "classes": [], "scores": []})
            img_path = os.path.join(img_dir, ginfo["file_name"])
            if not os.path.exists(img_path):
                continue

            tp_pairs, _, _ = match(ginfo["bboxes"], ginfo["classes"],
                                   pinfo["bboxes"],  pinfo["classes"])
            target_idx = {list(CLASSES).index(c) for c in TARGET_CLASSES if c in CLASSES}
            litter_tp  = sum(1 for gi, _ in tp_pairs
                             if ginfo["classes"][gi] in target_idx)

            out_name = f"tp{litter_tp:02d}_{os.path.basename(ginfo['file_name'])}"
            out_path = os.path.join(out_dir, out_name)
            title = (f"[{label}/{split}] {ginfo['video_name']}\n"
                     f"litter/garbage TP={litter_tp}")
            draw_frame(img_path,
                       ginfo["bboxes"], ginfo["classes"],
                       pinfo["bboxes"], pinfo["classes"], pinfo["scores"],
                       out_path, title)

        print(f"  [{label}/{split}] 完成 → {out_dir}（{len(tp_gt)} 張）")


def main():
    p = argparse.ArgumentParser(
        description="第七次 vs 第八次 litter/garbage TP frame 視覺化比較")
    p.add_argument("--v7-checkpoint", required=True,
                   help="第七次 best checkpoint .pth")
    p.add_argument("--v8-checkpoint", required=True,
                   help="第八次 best checkpoint .pth")
    p.add_argument("--v7-ea-dir", required=True,
                   help="第七次 error_analysis 目錄（含 {split}_per_frame_class_errors.csv）")
    p.add_argument("--v8-ea-dir", required=True,
                   help="第八次 error_analysis 目錄")
    p.add_argument("--output-dir", required=True,
                   help="輸出根目錄（會在下面建 7th/ 和 8th/ 子目錄）")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== 第七次實驗（7th_expt_v9_loss_compensate_36e）===")
    process_run(
        label="7th",
        config=V7_CONFIG,
        checkpoint=args.v7_checkpoint,
        data_root=V7_DATA_ROOT,
        ea_dir=args.v7_ea_dir,
        output_dir=args.output_dir,
    )

    print("\n=== 第八次實驗（chiayi_v10_36e_65vids）===")
    process_run(
        label="8th",
        config=V8_CONFIG,
        checkpoint=args.v8_checkpoint,
        data_root=V8_DATA_ROOT,
        ea_dir=args.v8_ea_dir,
        output_dir=args.output_dir,
    )

    print(f"\n全部完成。輸出目錄：{args.output_dir}")


if __name__ == "__main__":
    main()
