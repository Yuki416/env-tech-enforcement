"""
用 Co-DETR COCO 預訓練權重對抽幀結果做推論，產生 COCO JSON（供 Roboflow 匯入）
只保留 person / car / truck，confidence >= CONF_THRESH

用法（在 container 內執行）：
  # 預設（539 影片）
  docker exec co-detr-test python3 /workspace/project/pseudo_label.py

  # 指定新影片的抽幀目錄
  docker exec co-detr-test python3 /workspace/project/pseudo_label.py \
      --frames-dir  /workspace/project/test_data/day16_frames \
      --output-json /workspace/project/test_data/day16_pseudo_labels.json \
      --vis-dir     /workspace/project/test_data/day16_vis
"""

import argparse
import os
import json
import glob
from PIL import Image
from mmdet.apis import DetInferencer

# ── 預設路徑（container 內）───────────────────────────────────────────────────
DEFAULT_CONFIG     = "/opt/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_r50_8xb2_1x_coco.py"
DEFAULT_CHECKPOINT = "/workspace/project/models/co_dino_5scale_r50_1x_coco-7481f903.pth"
DEFAULT_FRAMES_DIR = "/workspace/project/test_data/539_frames"
DEFAULT_OUTPUT_JSON = "/workspace/project/test_data/539_pseudo_labels.json"
DEFAULT_VIS_DIR    = "/workspace/project/test_data/539_vis"

# ── COCO label 對應 ───────────────────────────────────────────────────────────
# motorcycle(3) 加入：嘉義場景常見的機車
# truck(7) 加入：夜間板條箱貨車外型，模型判成 truck 而非 car
COCO_LABEL_MAP = {
    0: {"id": 1, "name": "person"},
    2: {"id": 2, "name": "car"},
    3: {"id": 3, "name": "motorcycle"},
    7: {"id": 4, "name": "truck"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Co-DETR pseudo label generator")
    parser.add_argument("--config",      default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint",  default=DEFAULT_CHECKPOINT)
    parser.add_argument("--frames-dir",  default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--vis-dir",     default=DEFAULT_VIS_DIR)
    parser.add_argument("--conf-thresh", type=float, default=0.3)
    return parser.parse_args()


def main():
    import time

    args = parse_args()

    print("初始化 Co-DETR（COCO pretrain）...")
    inferencer = DetInferencer(model=args.config, weights=args.checkpoint, device="cuda:0")

    frames = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")))
    if not frames:
        raise FileNotFoundError(f"找不到 jpg 幀：{args.frames_dir}")
    total = len(frames)
    print(f"共 {total} 幀，開始推論...")

    os.makedirs(args.vis_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    coco_images      = []
    coco_annotations = []
    ann_id           = 1
    t_start          = time.time()

    for img_id, img_path in enumerate(frames, start=1):
        filename = os.path.basename(img_path)
        t_frame  = time.time()

        with Image.open(img_path) as img:
            w, h = img.size

        coco_images.append({"id": img_id, "file_name": filename, "width": w, "height": h})

        result = inferencer(img_path, out_dir=args.vis_dir, show=False, print_result=False)
        pred   = result["predictions"][0]

        kept = 0
        for label, score, bbox in zip(pred["labels"], pred["scores"], pred["bboxes"]):
            if label not in COCO_LABEL_MAP or score < args.conf_thresh:
                continue
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1
            coco_annotations.append({
                "id":          ann_id,
                "image_id":    img_id,
                "category_id": COCO_LABEL_MAP[label]["id"],
                "bbox":        [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                "area":        round(bw * bh, 2),
                "iscrowd":     0,
                "score":       round(float(score), 4),
            })
            ann_id += 1
            kept += 1

        elapsed   = time.time() - t_start
        frame_sec = time.time() - t_frame
        fps       = img_id / elapsed
        remaining = (total - img_id) / fps if fps > 0 else 0
        pct       = img_id / total * 100
        print(f"  [{img_id:4d}/{total}] {pct:5.1f}%  {filename}  "
              f"→ {kept} 框  {frame_sec:.2f}s/frame  avg {fps:.2f} fps  "
              f"剩 {int(remaining//60)}m{int(remaining%60):02d}s")

    coco_output = {
        "images":      coco_images,
        "annotations": coco_annotations,
        "categories":  [v for v in COCO_LABEL_MAP.values()],
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, indent=2, ensure_ascii=False)

    print(f"\n完成！")
    print(f"  圖片數：{len(coco_images)}")
    print(f"  標註數：{len(coco_annotations)}")
    print(f"  輸出 JSON：{args.output_json}")
    print(f"  視覺化：{args.vis_dir}/vis/")


if __name__ == "__main__":
    main()
