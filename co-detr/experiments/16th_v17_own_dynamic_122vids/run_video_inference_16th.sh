#!/usr/bin/env bash
# 第16次 v17 影片推論完整鏈（三個 split：train/valid/test）。
# Step 1: infer_16th_v17_video_classification.py --splits train valid test（整檔覆寫 csv/mp4）
# Step 2: fix_video_classifications_16th.py（IoU≥0.5 校正，test 用 full655_score03，train/valid 用 score03）
# 收尾：把產物 chmod 644（容器 root 寫出的檔預設會有 0600 的校正檔，主機帳號才讀得到→能上傳）
#
# 用法（背景）：
#   nohup bash experiments/16th_v17_own_dynamic_122vids/run_video_inference_16th.sh \
#     > /path/to/data/16th_expt_v17_own_dynamic_122vids_36e/video_inference_full.log 2>&1 &
#   tail -f /path/to/data/16th_expt_v17_own_dynamic_122vids_36e/video_inference_full.log
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # = co-detr 根目錄
EXP=experiments/16th_v17_own_dynamic_122vids
CONTAINER="co-detr-v17-videoinfer"

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
    python /workspace/project/${EXP}/infer_16th_v17_video_classification.py \
    --splits train valid test

echo "=== Step 2: IoU≥0.5 校正 ==="
docker exec "$CONTAINER" \
    python /workspace/project/${EXP}/fix_video_classifications_16th.py

echo "=== 收尾：chmod 644（讓主機帳號可讀→可上傳）==="
docker exec "$CONTAINER" bash -c \
    'chmod -R a+r /workspace/hdd2_storage/16th_expt_v17_own_dynamic_122vids_36e/video_inference && \
     find /workspace/hdd2_storage/16th_expt_v17_own_dynamic_122vids_36e/video_inference -type d -exec chmod a+rx {} +'

echo "VIDEO_INFERENCE_FULL_DONE"
