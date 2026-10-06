#!/usr/bin/env bash
# 第 10 次訓練 video inference：對 v12 dataset train/valid/test 跑 best epoch 17 推論，
# 逐 frame IoU>=0.5 配對，輸出原解析度帶框 mp4 + per-video/per-frame CSV + summary。
#
# 用法： bash run_infer_10th_v12_litter_video.sh
# 輸出： /path/to/data/10th_expt_v12_other_team_litter_36e/video_inference/

set -euo pipefail

PROJECT_DIR="/path/to/repo/co-detr"
HDD_DIR="/path/to/data"
LOG_FILE="/path/to/data/10th_expt_v12_other_team_litter_36e/video_inference.log"

mkdir -p "$(dirname "$LOG_FILE")"

echo "啟動 co-detr video inference (GPU 0)..."
echo "Log: ${LOG_FILE}"

nohup docker run --rm \
    --name co-detr-infer-10th \
    --gpus '"device=0"' \
    --ipc host \
    --shm-size 8g \
    -v "${PROJECT_DIR}:/workspace/project" \
    -v "${HDD_DIR}:${HDD_DIR}" \
    -e PYTHONPATH=/opt/mmdetection:/workspace/project \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e LANG=C.UTF-8 \
    -e LC_ALL=C.UTF-8 \
    co-detr:latest \
    python /workspace/project/infer_10th_v12_litter_video_classification.py \
        --splits test valid train \
        --device cuda:0 \
    > "${LOG_FILE}" 2>&1 &

echo "PID: $!"
echo "追蹤： tail -f ${LOG_FILE}"
