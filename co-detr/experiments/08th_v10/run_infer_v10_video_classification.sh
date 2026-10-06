#!/usr/bin/env bash
# 對 v10 dataset 全部影片跑 epoch_7 inference，輸出帶 bbox 的 MP4 + TP/TN/FP/FN 分類
#
# 用法：
#   bash run_infer_v10_video_classification.sh
#
# 輸出：/path/to/data/chiayi_v10_36e_65vids/video_inference/
#   train/*.mp4, valid/*.mp4, test/*.mp4
#   video_classification.csv
#   video_classification_summary.md

set -euo pipefail

PROJECT_DIR="/path/to/repo/co-detr"
HDD_DIR="/path/to/data"
LOG_FILE="${PROJECT_DIR}/results/v10_video_inference.log"

echo "啟動 co-detr inference (GPU 1)..."
echo "Log: ${LOG_FILE}"
echo "輸出: ${HDD_DIR}/user/chiayi_v10_36e_65vids/video_inference/"
echo ""

nohup docker run --rm \
    --name co-detr-test \
    --gpus '"device=1"' \
    --ipc host \
    --shm-size 8g \
    -v "${PROJECT_DIR}:/workspace/project" \
    -v "${HDD_DIR}:${HDD_DIR}" \
    -e PYTHONPATH=/opt/mmdetection:/workspace/project \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e LANG=C.UTF-8 \
    -e LC_ALL=C.UTF-8 \
    co-detr:latest \
    python /workspace/project/infer_v10_video_classification.py \
        --splits train valid test \
        --device cuda:0 \
    > "${LOG_FILE}" 2>&1 &

echo "PID: $!"
echo ""
echo "追蹤進度："
echo "  tail -f ${LOG_FILE}"
