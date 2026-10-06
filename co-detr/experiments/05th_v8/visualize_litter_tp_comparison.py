"""
產生 litter/garbage TP frame 的視覺化比較圖。
輸出結構：
  litter_tp_comparison/
    v8/train/  v8/valid/  v8/test/
    v9/train/  v9/valid/  v9/test/
"""
import os, csv, argparse
from collections import defaultdict
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D

# ── 共用常數 ──────────────────────────────────────────────────────────────────
V8_CLASSES   = ("car", "litter", "motorcycle", "person")
V9_CLASSES   = ("car", "litter", "motorcycle", "person", "garbage")
V8_TARGET    = {"litter"}
V9_TARGET    = {"litter", "garbage"}

V8_EA_DIR    = "/workspace/project/results/chiayi_v8_dense_36e/error_analysis"
V9_EA_DIR    = "/workspace/project/results/chiayi_v9_36e/error_analysis"

V8_DATA_ROOT = "/workspace/hdd2_storage/coco/chiayi_v8_dense_annotation"
V9_DATA_ROOT = "/workspace/datasets/coco/chiayi_v9_garbage_litter"

V8_IMG_DIRS  = {"train": "train", "valid": "valid", "test": "test"}
V9_IMG_DIRS  = {"train": "train", "valid": "valid", "test": "test"}

V8_ANN       = {"train": "annotations/train.json",
                "valid": "annotations/valid.json",
                "test":  "annotations/test.json"}
V9_ANN       = {"train": "annotations/train.json",
                "valid": "annotations/valid.json",
                "test":  "annotations/test.json"}

OUTPUT_DIR   = "/workspace/project/results/litter_tp_comparison"

# ── 推論 ──────────────────────────────────────────────────────────────────────
def run_inference(config, checkpoint, img_dir, gt_data, score_thr=0.3):
    from mmdet.apis import init_detector, inference_detector
    model = init_detector(config, checkpoint, device="cuda:0")
    pred = {}
    files = list(gt_data.keys())
    for i, img_id in enumerate(files):
        if i % 200 == 0:
            print(f"  推論 {i}/{len(files)} ...")
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
                if sc < score_thr:
                    continue
                x1, y1, x2, y2 = inst.bboxes[j].tolist()
                bboxes.append([x1, y1, x2 - x1, y2 - y1])
                classes.append(int(inst.labels[j]))
                scores.append(sc)
        pred[img_id] = {"bboxes": bboxes, "classes": classes, "scores": scores}
    return pred

def load_coco_gt(json_path):
    import json
    with open(json_path, encoding="utf-8") as f:
        d = json.load(f)
    cat_map = {c["id"]: c["name"] for c in d["categories"]}
    cls_list = [c["name"] for c in d["categories"]]
    img_map  = {img["id"]: img for img in d["images"]}
    gt = {}
    for img in d["images"]:
        fn = img["file_name"]
        base = os.path.splitext(fn)[0]
        video = base.rsplit("_frame_", 1)[0] if "_frame_" in base else base
        gt[img["id"]] = {"file_name": fn, "video_name": video,
                         "bboxes": [], "classes": []}
    for ann in d["annotations"]:
        iid = ann["image_id"]
        if iid not in gt:
            continue
        name = cat_map[ann["category_id"]]
        if name in cls_list:
            gt[iid]["bboxes"].append(ann["bbox"])
            gt[iid]["classes"].append(cls_list.index(name))
    return gt, cls_list

# ── IoU ──────────────────────────────────────────────────────────────────────
def iou_xywh(a, b):
    ax1, ay1 = a[0], a[1]; ax2, ay2 = a[0]+a[2], a[1]+a[3]
    bx1, by1 = b[0], b[1]; bx2, by2 = b[0]+b[2], b[1]+b[3]
    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter / union if union > 0 else 0.0

def match(gt_boxes, gt_cls, pred_boxes, pred_cls, iou_thr=0.5):
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
        if best_iou >= iou_thr:
            matched_gt.add(best_gi); matched_pred.add(pi)
            tp_pairs.append((best_gi, pi))
    fn = [gi for gi in range(len(gt_boxes)) if gi not in matched_gt]
    fp = [pi for pi in range(len(pred_boxes)) if pi not in matched_pred]
    return tp_pairs, fn, fp

# ── 視覺化 ────────────────────────────────────────────────────────────────────
def draw_frame(img_path, gt_boxes, gt_cls, pred_boxes, pred_cls, pred_scores,
               classes, target_classes, out_path, title=""):
    img = np.array(Image.open(img_path).convert("RGB"))
    h, w = img.shape[:2]
    fig, ax = plt.subplots(1, figsize=(w/120, h/120))
    ax.imshow(img)

    tp_pairs, fn_idx, fp_idx = match(gt_boxes, gt_cls, pred_boxes, pred_cls)
    tp_gt = {gi for gi, _ in tp_pairs}
    tp_pred = {pi for _, pi in tp_pairs}

    target_idx = {classes.index(c) for c in target_classes if c in classes}

    for gi, pi in tp_pairs:
        if gt_cls[gi] not in target_idx:
            continue
        # GT 框（綠）
        x, y, bw, bh = gt_boxes[gi]
        ax.add_patch(patches.Rectangle((x,y), bw, bh, lw=2,
                                        edgecolor="#00cc00", facecolor="none"))
        ax.text(x, max(0,y-4), f"GT:{classes[gt_cls[gi]]}",
                color="#00cc00", fontsize=9, fontweight="bold", va="bottom",
                bbox=dict(facecolor="white", alpha=0.6, pad=1, linewidth=0))
        # Pred 框（藍）
        x, y, bw, bh = pred_boxes[pi]
        ax.add_patch(patches.Rectangle((x,y), bw, bh, lw=2,
                                        edgecolor="#2288ff", facecolor="none"))
        ax.text(x, min(h, y+bh+4), f"TP:{classes[pred_cls[pi]]}({pred_scores[pi]:.2f})",
                color="#2288ff", fontsize=9, fontweight="bold", va="top",
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
        ax.text(x, max(0,y-4), f"FN:{classes[gt_cls[gi]]}",
                color="red", fontsize=9, va="bottom",
                bbox=dict(facecolor="white", alpha=0.6, pad=1, linewidth=0))

    legend = [
        Line2D([0],[0], color="#00cc00", lw=2, label="GT (matched)"),
        Line2D([0],[0], color="#2288ff", lw=2, label="TP prediction"),
        Line2D([0],[0], color="red",    lw=2, linestyle="--", label="FN (missed GT)"),
    ]
    ax.legend(handles=legend, loc="upper right", fontsize=8)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()

# ── 主流程 ────────────────────────────────────────────────────────────────────
def process_run(label, config, checkpoint, ann_dict, img_dir_dict,
                ea_dir, target_classes, classes, data_root):
    for split in ["train", "valid", "test"]:
        csv_path = os.path.join(ea_dir, f"{split}_per_frame_class_errors.csv")
        if not os.path.exists(csv_path):
            print(f"  [{label}/{split}] 找不到 {csv_path}，跳過")
            continue

        # 找出有 litter/garbage TP 的 frame
        tp_frames = set()
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row["class"] in target_classes and int(row["TP"]) > 0:
                    tp_frames.add(row["file_name"])

        if not tp_frames:
            print(f"  [{label}/{split}] 無 litter/garbage TP，跳過")
            continue

        print(f"  [{label}/{split}] 有 TP 的 frame：{len(tp_frames)} 張，載入 GT ...")
        gt_data, _ = load_coco_gt(os.path.join(data_root, ann_dict[split]))
        # 只保留有 TP 的 frame
        tp_gt = {iid: v for iid, v in gt_data.items()
                 if v["file_name"] in tp_frames}

        print(f"  [{label}/{split}] 執行推論 ...")
        img_dir = os.path.join(data_root, img_dir_dict[split])
        pred = run_inference(config, checkpoint, img_dir, tp_gt)

        out_dir = os.path.join(OUTPUT_DIR, label, split)
        os.makedirs(out_dir, exist_ok=True)

        print(f"  [{label}/{split}] 產生視覺化 ...")
        for iid, ginfo in tp_gt.items():
            pinfo = pred.get(iid, {"bboxes":[],"classes":[],"scores":[]})
            img_path = os.path.join(img_dir, ginfo["file_name"])
            if not os.path.exists(img_path):
                continue
            tp_pairs, _, _ = match(ginfo["bboxes"], ginfo["classes"],
                                   pinfo["bboxes"],  pinfo["classes"])
            litter_tp = sum(1 for gi,_ in tp_pairs
                            if classes[ginfo["classes"][gi]] in target_classes)
            out_name = f"tp{litter_tp:02d}_{os.path.basename(ginfo['file_name'])}"
            out_path = os.path.join(out_dir, out_name)
            title = f"[{label}/{split}] {ginfo['video_name']}\nlitter/garbage TP={litter_tp}"
            draw_frame(img_path, ginfo["bboxes"], ginfo["classes"],
                       pinfo["bboxes"], pinfo["classes"], pinfo["scores"],
                       classes, target_classes, out_path, title)
        print(f"  [{label}/{split}] 完成 → {out_dir}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=== v8（第四次）===")
    process_run(
        label="v8",
        config="/workspace/project/configs/co_dino_chiayi_v8_dense_36e.py",
        checkpoint="/workspace/project/results/chiayi_v8_dense_36e/best_coco_car_precision_epoch_8.pth",
        ann_dict=V8_ANN, img_dir_dict=V8_IMG_DIRS,
        ea_dir=V8_EA_DIR, target_classes=V8_TARGET,
        classes=list(V8_CLASSES), data_root=V8_DATA_ROOT,
    )
    print("\n=== v9（第六次）===")
    process_run(
        label="v9",
        config="/workspace/project/configs/co_dino_chiayi_v9_36e.py",
        checkpoint="/workspace/project/results/chiayi_v9_36e/best_coco_car_precision_epoch_2.pth",
        ann_dict=V9_ANN, img_dir_dict=V9_IMG_DIRS,
        ea_dir=V9_EA_DIR, target_classes=V9_TARGET,
        classes=list(V9_CLASSES), data_root=V9_DATA_ROOT,
    )
    print(f"\n全部完成。輸出目錄：{OUTPUT_DIR}")

if __name__ == "__main__":
    main()
