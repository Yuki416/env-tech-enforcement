#!/usr/bin/env bash
# 第18次 v18 BGS 4ch 影片推論完整鏈（train/valid/test 三 split）。
# infer_18th_v18_4ch_video_classification.py：4ch 推論（LoadImageAndChangemap 讀 changemap）+
#   逐 clip bbox mp4 + IoU≥0.5 **內建**校正的 clip 分類（免另跑 fix / 4ch error_analysis）。
# 收尾 chmod 644（容器 root 寫出的檔主機才讀得到→能上傳）。
# 用法（背景）：nohup bash .../run_video_inference_18th.sh > .../video_inference_full.log 2>&1 & ; tail -f
set -euo pipefail
PROJ=/path/to/repo/co-detr
EXP=experiments/18th_v18_own_dynamic_bgs_122vids
CONTAINER=co-detr-v18-videoinfer
RESDIR=/path/to/data/18th_expt_v18_own_dynamic_bgs_4ch_36e

docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --name "$CONTAINER" --gpus all --ipc=host --shm-size=16g \
  -v "$PROJ":/workspace/project \
  -v /path/to/data/coco:/workspace/datasets/coco \
  -v /path/to/data:/workspace/hdd2_storage \
  -e PYTHONPATH=/opt/mmdetection:/workspace/project \
  -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 -e PYTHONUTF8=1 \
  co-detr:latest sleep infinity
trap "docker rm -f $CONTAINER 2>/dev/null || true" EXIT

echo "=== 4ch 影片推論（train/valid/test 三 split，IoU≥0.5 內建校正）==="
docker exec -e CUDA_VISIBLE_DEVICES=0 "$CONTAINER" \
  python /workspace/project/${EXP}/infer_18th_v18_4ch_video_classification.py --splits train valid test

echo "=== 收尾：chmod 644（讓主機帳號可讀→可上傳）==="
docker exec "$CONTAINER" bash -c \
  'chmod -R a+r /workspace/hdd2_storage/18th_expt_v18_own_dynamic_bgs_4ch_36e/video_inference && \
   find /workspace/hdd2_storage/18th_expt_v18_own_dynamic_bgs_4ch_36e/video_inference -type d -exec chmod a+rx {} +'

echo "VIDEO_INFERENCE_FULL_DONE"
