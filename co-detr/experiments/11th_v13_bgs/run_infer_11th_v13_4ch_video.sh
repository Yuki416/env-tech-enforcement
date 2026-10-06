#!/usr/bin/env bash
# 第 11 次訓練 video inference（v13 BGS 4ch）：對 v13 dataset train/valid/test 跑 best ckpt 推論，
# 逐 frame IoU>=0.5 配對，輸出原解析度帶框 mp4 + per-video/per-frame CSV + summary。
#
# ★ 模型是 4 通道（RGB+changemap）；推論腳本用 init_detector + 自訂 4ch pipeline（非 DetInferencer）。
#
# 用法： bash run_infer_11th_v13_4ch_video.sh
# 輸出： /path/to/data/11th_expt_v13_bgs_4ch_litter_36e/video_inference/

set -euo pipefail

PROJECT_DIR="/path/to/repo/co-detr"
HDD_DIR="/path/to/data"
LOG_FILE="/path/to/data/11th_expt_v13_bgs_4ch_litter_36e/video_inference.log"

mkdir -p "$(dirname "$LOG_FILE")"

echo "啟動 co-detr 4ch video inference (GPU 0)..."
echo "Log: ${LOG_FILE}"

nohup docker run --rm \
    --name co-detr-infer-11th \
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
    python /workspace/project/infer_11th_v13_4ch_litter_video_classification.py \
        --splits test valid train \
        --device cuda:0 \
    > "${LOG_FILE}" 2>&1 &

echo "PID: $!"
echo "追蹤： tail -f ${LOG_FILE}"
