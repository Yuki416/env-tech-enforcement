"""
批次對 午後_clips_3 做推論，產生 pseudo labels。
- Day_Garbage → 7 fps；Day_Litter → 10 fps
- Confidence threshold: 0.3
- 不做 Car/Truck cross-class NMS（保留所有框）

用法（container 內）：
  docker exec co-detr-test python3 /workspace/project/batch_infer_午後_no_suppress.py
"""

import os
import glob
import json
import argparse
import subprocess
import time
from PIL import Image
from mmdet.apis import DetInferencer

parser = argparse.ArgumentParser()
parser.add_argument("--video-dir",  default="/workspace/hdd2_storage/午後_clips_3")
parser.add_argument("--output-dir", default="/workspace/hdd2_storage/pseudo_labels_午後_no_suppress")
parser.add_argument("--conf-thresh", type=float, default=0.3)
args = parser.parse_args()

VIDEO_DIR  = args.video_dir
OUTPUT_DIR = args.output_dir
CONF_THRESH = args.conf_thresh

CONFIG     = "/opt/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_r50_8xb2_1x_coco.py"
CHECKPOINT = "/workspace/project/models/co_dino_5scale_r50_1x_coco-7481f903.pth"

COCO_LABEL_MAP = {
    0: {"id": 1, "name": "person"},
    2: {"id": 2, "name": "car"},
    3: {"id": 3, "name": "motorcycle"},
    7: {"id": 4, "name": "truck"},
}

CAR_ID   = 2
TRUCK_ID = 4


def compute_iou(box_a, box_b):
    """box 格式：[x1, y1, w, h]（COCO）"""
    ax1, ay1, aw, ah = box_a
    bx1, by1, bw, bh = box_b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union


def suppress_car_truck_overlap(anns, iou_thresh):
    """
    同一幀的 annotations 裡，若某個 car 框和某個 truck 框 IoU > iou_thresh，
    移除 score 較低的那個。回傳過濾後的 list。
    """
    cars   = [a for a in anns if a["category_id"] == CAR_ID]
    trucks = [a for a in anns if a["category_id"] == TRUCK_ID]
    others = [a for a in anns if a["category_id"] not in (CAR_ID, TRUCK_ID)]

    suppress = set()
    for c in cars:
        for t in trucks:
            if compute_iou(c["bbox"], t["bbox"]) > iou_thresh:
                # 移除 score 較低的
                loser = c if c["score"] < t["score"] else t
                suppress.add(id(loser))

    kept = [a for a in (cars + trucks) if id(a) not in suppress]
    return others + kept


# ── 初始化模型 ────────────────────────────────────────────────────────────────
print("初始化 Co-DETR...")
inferencer = DetInferencer(model=CONFIG, weights=CHECKPOINT, device="cuda:0")
print(f"模型載入完成  conf={CONF_THRESH}  (no car/truck suppress)\n")

video_paths = sorted(
    p for p in glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
)
print(f"共找到 {len(video_paths)} 部影片\n")
os.makedirs(OUTPUT_DIR, exist_ok=True)

for vid_idx, video_path in enumerate(video_paths, start=1):
    video_file = os.path.basename(video_path)
    video_name = os.path.splitext(video_file)[0]
    out_dir     = os.path.join(OUTPUT_DIR, video_name)
    frames_dir  = os.path.join(out_dir, "frames")
    vis_dir     = os.path.join(out_dir, "vis")
    output_json = os.path.join(out_dir, "pseudo_labels.json")

    fps = 7 if "Garbage" in video_file else 10

    print(f"[{vid_idx:2d}/{len(video_paths)}] {video_name}  fps={fps}")

    if os.path.exists(output_json):
        print(f"  → 已存在，跳過\n")
        continue

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # 抽幀
    print(f"  抽幀中...")
    t0 = time.time()
    result = subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}", "-q:v", "2",
        os.path.join(frames_dir, "frame_%04d.jpg"),
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ✗ ffmpeg 失敗：{result.stderr[-200:]}\n")
        continue

    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    print(f"  抽出 {len(frames)} 幀（{time.time()-t0:.1f}s）")

    # 推論
    print(f"  推論中...")
    coco_images      = []
    coco_annotations = []
    ann_id  = 1
    t_start = time.time()

    for img_id, img_path in enumerate(frames, start=1):
        filename = os.path.basename(img_path)
        t_frame  = time.time()

        with Image.open(img_path) as img:
            w, h = img.size
        coco_images.append({"id": img_id, "file_name": filename, "width": w, "height": h})

        pred_result = inferencer(img_path, out_dir=vis_dir, show=False, print_result=False)
        pred        = pred_result["predictions"][0]

        # 收集符合 conf 的框
        frame_anns = []
        for label, score, bbox in zip(pred["labels"], pred["scores"], pred["bboxes"]):
            if label not in COCO_LABEL_MAP or score < CONF_THRESH:
                continue
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1
            frame_anns.append({
                "id":          None,
                "image_id":    img_id,
                "category_id": COCO_LABEL_MAP[label]["id"],
                "bbox":        [round(x1, 2), round(y1, 2), round(bw, 2), round(bh, 2)],
                "area":        round(bw * bh, 2),
                "iscrowd":     0,
                "score":       round(float(score), 4),
            })

        for ann in frame_anns:
            ann["id"] = ann_id
            ann_id += 1
        coco_annotations.extend(frame_anns)

        elapsed   = time.time() - t_start
        fps_avg   = img_id / elapsed
        remaining = (len(frames) - img_id) / fps_avg if fps_avg > 0 else 0
        print(f"    [{img_id:4d}/{len(frames)}] {filename}"
              f"  → {len(frame_anns)} 框  {time.time()-t_frame:.2f}s"
              f"  剩 {int(remaining//60)}m{int(remaining%60):02d}s")

    coco_output = {
        "images":      coco_images,
        "annotations": coco_annotations,
        "categories":  [v for v in COCO_LABEL_MAP.values()],
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(coco_output, f, indent=2, ensure_ascii=False)

    total_sec = time.time() - t_start
    print(f"  ✓ 完成  圖片:{len(coco_images)}  標註:{len(coco_annotations)}"
          f"  耗時:{int(total_sec//60)}m{int(total_sec%60):02d}s\n")

print("=== 全部完成 ===")
