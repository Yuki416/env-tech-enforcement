#!/usr/bin/env bash
# 變因① statictail — Video Inference 完整鏈（train/valid/test）。
# ★ 前置：post_analysis_var1.sh（Error Analysis）必須先跑完 —— Step 2 fix 要吃它的 per-frame CSV。
# Step 1: infer_var1_...py（畫框 + 粗分類）
# Step 2: fix_video_classifications_var1.py（讀 error_analysis，IoU≥0.5 校正）
# 收尾: chmod 644（容器 root 寫的檔，主機才讀得到→可上傳）
# 用法（背景）：
#   nohup bash experiments/ablation_study_4vars_20260804/run_video_inference_var1.sh \
#     > /path/to/data/ablation_var1_statictail_36e/video_inference.log 2>&1 & ; tail -f ...
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXP=experiments/ablation_study_4vars_20260804
RESULT=/workspace/hdd2_storage/ablation_var1_statictail_36e
HOSTRES=/path/to/data/ablation_var1_statictail_36e
CONTAINER="co-detr-var1-videoinfer"

# 前置檢查（host 端）
ls ${HOSTRES}/best_coco_litter_precision_epoch_*.pth >/dev/null 2>&1 || { echo "ERROR: 無 best ckpt，訓練完成了嗎？"; exit 1; }
for f in error_analysis_score03/train_per_frame_class_errors.csv \
         error_analysis_score03/valid_per_frame_class_errors.csv \
         error_analysis_full655_score03/test_per_frame_class_errors.csv; do
  [ -f "${HOSTRES}/$f" ] || { echo "ERROR: 缺 ${HOSTRES}/$f → 請先跑 post_analysis_var1.sh（Error Analysis）"; exit 1; }
done

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

echo "=== Step 1: 影片推論 + 粗分類（train/valid/test）==="
docker exec -e CUDA_VISIBLE_DEVICES=0 "$CONTAINER" \
    python /workspace/project/${EXP}/infer_var1_statictail_video_classification.py --splits train valid test

echo "=== Step 2: IoU≥0.5 校正（讀 error_analysis per-frame CSV）==="
docker exec "$CONTAINER" \
    python /workspace/project/${EXP}/fix_video_classifications_var1.py

echo "=== 收尾：chmod 644 ==="
docker exec "$CONTAINER" bash -c \
    "chmod -R a+r ${RESULT}/video_inference && find ${RESULT}/video_inference -type d -exec chmod a+rx {} +"

echo "VIDEO_INFERENCE_VAR1_DONE"
