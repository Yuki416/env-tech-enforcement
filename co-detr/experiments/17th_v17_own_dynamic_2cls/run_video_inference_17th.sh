#!/usr/bin/env bash
# 第17次 v17 影片推論完整鏈（2 類 litter+garbage；三個 split：train/valid/test）。
# Step 1: infer_17th_v17_2cls_video_classification.py --splits train valid test（整檔覆寫 csv/mp4）
# Step 2: fix_video_classifications_17th.py（IoU≥0.5 校正，test 用 full655_score03，train/valid 用 score03）
# 收尾：把產物 chmod（容器 root 寫出的檔預設 0600，主機帳號才讀得到→能上傳/瀏覽）
#
# 前置：post_analysis_17th_v17_2cls.sh 需已跑完（Step 2 要吃 error_analysis_score03 / full655_score03）。
#
# 用法（背景）：
#   nohup bash experiments/17th_v17_own_dynamic_2cls/run_video_inference_17th.sh \
#     > /path/to/data/17th_expt_v17_own_dynamic_2cls_36e/video_inference_full.log 2>&1 &
#   tail -f /path/to/data/17th_expt_v17_own_dynamic_2cls_36e/video_inference_full.log
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # = co-detr 根目錄
EXP=experiments/17th_v17_own_dynamic_2cls
CONTAINER="co-detr-v17-2cls-videoinfer"
RESULT_DIR=/workspace/hdd2_storage/17th_expt_v17_own_dynamic_2cls_36e

docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --name "$CONTAINER" \
    --gpus all --ipc=host --shm-size=16g \
    -v "${SCRIPT_DIR}:/workspace/project" \
    -v /path/to/data/coco:/workspace/datasets/coco \
    -v /path/to/data:/workspace/hdd2_storage \
    -e PYTHONPATH=/opt/mmdetection:/workspace/project \
    -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 -e PYTHONUTF8=1 \
    co-detr:latest sleep infinity
trap "echo 'Stopping container...'; docker rm -f $CONTAINER 2>/dev/null || true" EXIT

echo "=== Step 1: 影片推論（train/valid/test 三 split）==="
docker exec -e CUDA_VISIBLE_DEVICES=0 "$CONTAINER" \
    python /workspace/project/${EXP}/infer_17th_v17_2cls_video_classification.py \
    --splits train valid test

echo "=== Step 2: IoU≥0.5 校正 ==="
docker exec "$CONTAINER" \
    python /workspace/project/${EXP}/fix_video_classifications_17th.py

echo "=== 收尾：chmod（讓主機帳號可讀→可上傳）==="
docker exec "$CONTAINER" bash -c \
    "chmod -R a+r ${RESULT_DIR}/video_inference && \
     find ${RESULT_DIR}/video_inference -type d -exec chmod a+rx {} +"

echo "VIDEO_INFERENCE_FULL_DONE"
