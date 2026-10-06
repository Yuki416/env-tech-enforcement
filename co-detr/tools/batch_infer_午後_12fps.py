"""
針對指定的兩部影片用 12 fps 重抽 + inference（取代原 10 fps）。
- conf >= 0.3，4 class：person/car/motorcycle/truck
- 不做 suppress（與 no_suppress 一致）
- 輸出覆蓋 pseudo_labels_午後_no_suppress 對應資料夾

用法（container 內）：
  docker exec/run ... python3 /workspace/project/batch_infer_午後_12fps.py
"""
import os, glob, json, subprocess, time
from PIL import Image
from mmdet.apis import DetInferencer

VIDEO_DIR  = "/workspace/hdd2_storage/午後_clips_3"
OUTPUT_DIR = "/workspace/hdd2_storage/pseudo_labels_午後_no_suppress"
FPS = 12
CONF_THRESH = 0.3

VIDEOS = [
    "Day_Litter_538_8_loc538_20240814165318_01m35s",
    "Day_Litter_538_8_loc538_20240823152858_04m28s",
]

CONFIG     = "/opt/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_r50_8xb2_1x_coco.py"
CHECKPOINT = "/workspace/project/models/co_dino_5scale_r50_1x_coco-7481f903.pth"
COCO_LABEL_MAP = {
    0: {"id": 1, "name": "person"},
    2: {"id": 2, "name": "car"},
    3: {"id": 3, "name": "motorcycle"},
    7: {"id": 4, "name": "truck"},
}

print("初始化 Co-DETR...")
inferencer = DetInferencer(model=CONFIG, weights=CHECKPOINT, device="cuda:0")
print(f"模型載入完成  fps={FPS}  conf={CONF_THRESH}\n")

for vid_idx, video_name in enumerate(VIDEOS, start=1):
    video_path  = os.path.join(VIDEO_DIR, video_name + ".mp4")
    out_dir     = os.path.join(OUTPUT_DIR, video_name)
    frames_dir  = os.path.join(out_dir, "frames")
    vis_dir     = os.path.join(out_dir, "vis")
    output_json = os.path.join(out_dir, "pseudo_labels.json")

    print(f"[{vid_idx}/{len(VIDEOS)}] {video_name}  fps={FPS}")
    if not os.path.exists(video_path):
        print(f"  ✗ 找不到影片：{video_path}\n")
        continue

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    print("  抽幀中...")
    t0 = time.time()
    r = subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={FPS}", "-q:v", "2",
        os.path.join(frames_dir, "frame_%04d.jpg"),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  ✗ ffmpeg 失敗：{r.stderr[-200:]}\n")
        continue

    frames = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    print(f"  抽出 {len(frames)} 幀（{time.time()-t0:.1f}s）")

    coco_images, coco_annotations = [], []
    ann_id = 1
    t_start = time.time()

    for img_id, img_path in enumerate(frames, start=1):
        filename = os.path.basename(img_path)
        with Image.open(img_path) as im:
            w, h = im.size
        coco_images.append({"id": img_id, "file_name": filename, "width": w, "height": h})

        pred = inferencer(img_path, out_dir=vis_dir, show=False, print_result=False)["predictions"][0]
        for label, score, bbox in zip(pred["labels"], pred["scores"], pred["bboxes"]):
            if label not in COCO_LABEL_MAP or score < CONF_THRESH:
                continue
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1
            coco_annotations.append({
                "id": ann_id, "image_id": img_id,
                "category_id": COCO_LABEL_MAP[label]["id"],
                "bbox": [round(x1,2), round(y1,2), round(bw,2), round(bh,2)],
                "area": round(bw*bh,2), "iscrowd": 0,
                "score": round(float(score),4),
            })
            ann_id += 1

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"images": coco_images, "annotations": coco_annotations,
                   "categories": [v for v in COCO_LABEL_MAP.values()]},
                  f, indent=2, ensure_ascii=False)
    print(f"  ✓ 完成  圖片:{len(coco_images)}  標註:{len(coco_annotations)}"
          f"  耗時:{int((time.time()-t_start)//60)}m{int((time.time()-t_start)%60):02d}s\n")

print("=== 全部完成 ===")
