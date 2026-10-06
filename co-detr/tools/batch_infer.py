"""
批次對指定資料夾底下所有影片做推論（支援遞迴掃描子資料夾）
- Garbage 影片：7 fps 抽幀
- Litter  影片：10 fps 抽幀
- 4 個 class：person / car / motorcycle / truck
- 輸出至指定 OUTPUT_DIR/{影片名稱（去副檔名）}/
  - frames/             ← 抽幀結果
  - pseudo_labels.json  ← COCO 格式標註
  - vis/                ← 視覺化結果

用法（在 container 內執行）：
  docker exec co-detr-test python3 /workspace/project/batch_infer.py \\
      --video-dir /workspace/hdd2_storage/violation_clips_systematic/深夜 \\
      --output-dir /workspace/hdd2_storage/pseudo_labels_深夜

支援中斷續跑：若 pseudo_labels.json 已存在則跳過該影片。
"""

import os
import glob
import json
import argparse
import subprocess
import time
from PIL import Image
from mmdet.apis import DetInferencer

# ── 參數解析 ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--video-dir",  default="/workspace/hdd2_storage/violation_clips_systematic/晚間_清晨")
parser.add_argument("--output-dir", default="/workspace/hdd2_storage/pseudo_labels")
args = parser.parse_args()

VIDEO_DIR  = args.video_dir
OUTPUT_DIR = args.output_dir

CONFIG     = "/opt/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_r50_8xb2_1x_coco.py"
CHECKPOINT = "/workspace/project/models/co_dino_5scale_r50_1x_coco-7481f903.pth"

CONF_THRESH = 0.3
COCO_LABEL_MAP = {
    0: {"id": 1, "name": "person"},
    2: {"id": 2, "name": "car"},
    3: {"id": 3, "name": "motorcycle"},
    7: {"id": 4, "name": "truck"},
}

# ── 初始化模型（只做一次）───────────────────────────────────────────────────
print("初始化 Co-DETR...")
inferencer = DetInferencer(model=CONFIG, weights=CHECKPOINT, device="cuda:0")
print("模型載入完成\n")

# ── 掃描影片（遞迴掃描子資料夾）───────────────────────────────────────────
video_paths = sorted(
    p for p in glob.glob(os.path.join(VIDEO_DIR, "**", "*"), recursive=True)
    if p.endswith(".mp4") or p.endswith(".avi")
)
print(f"共找到 {len(video_paths)} 部影片\n")

os.makedirs(OUTPUT_DIR, exist_ok=True)

for vid_idx, video_path in enumerate(video_paths, start=1):
    video_file  = os.path.basename(video_path)
    video_name  = os.path.splitext(video_file)[0]
    out_dir     = os.path.join(OUTPUT_DIR, video_name)
    frames_dir  = os.path.join(out_dir, "frames")
    vis_dir     = os.path.join(out_dir, "vis")
    output_json = os.path.join(out_dir, "pseudo_labels.json")

    fps = 7 if "Garbage" in video_file else 10

    print(f"[{vid_idx:2d}/{len(video_paths)}] {video_name}")
    print(f"  fps={fps}")

    # ── 跳過已完成的影片 ────────────────────────────────────────────────────
    if os.path.exists(output_json):
        print(f"  → 已存在 pseudo_labels.json，跳過\n")
        continue

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # ── 抽幀 ────────────────────────────────────────────────────────────────
    print(f"  抽幀中（{fps} fps）...")
    t0 = time.time()
    result = subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        os.path.join(frames_dir, "frame_%04d.jpg"),
    ], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ✗ ffmpeg 失敗：{result.stderr[-200:]}\n")
        continue

    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    print(f"  抽出 {len(frames)} 幀（{time.time()-t0:.1f}s）")

    # ── 推論 ────────────────────────────────────────────────────────────────
    print(f"  開始推論...")
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

        pred_result = inferencer(img_path, out_dir=vis_dir, show=False, print_result=False)
        pred        = pred_result["predictions"][0]

        kept = 0
        for label, score, bbox in zip(pred["labels"], pred["scores"], pred["bboxes"]):
            if label not in COCO_LABEL_MAP or score < CONF_THRESH:
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
            kept   += 1

        elapsed   = time.time() - t_start
        fps_avg   = img_id / elapsed
        remaining = (len(frames) - img_id) / fps_avg if fps_avg > 0 else 0
        pct       = img_id / len(frames) * 100
        print(f"    [{img_id:4d}/{len(frames)}] {pct:5.1f}%  {filename}"
              f"  → {kept} 框  {time.time()-t_frame:.2f}s  avg {fps_avg:.2f} fps"
              f"  剩 {int(remaining//60)}m{int(remaining%60):02d}s")

    # ── 輸出 JSON ───────────────────────────────────────────────────────────
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
