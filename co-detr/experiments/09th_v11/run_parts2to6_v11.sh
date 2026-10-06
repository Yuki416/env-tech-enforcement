#!/usr/bin/env bash
# Parts 2~6 only（Parts 1a/1b 資料已存在，跳過）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER="co-detr-v11-post"

docker run -d --name "$CONTAINER" \
    --gpus all --ipc=host --shm-size=16g \
    -v "${SCRIPT_DIR}:/workspace/project" \
    -v /path/to/data/coco:/workspace/datasets/coco \
    -v /path/to/data:/workspace/hdd2_storage \
    -e PYTHONPATH=/opt/mmdetection:/workspace/project \
    -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 \
    -e PYTHONUTF8=1 \
    co-detr:latest \
    sleep infinity

trap "echo 'Stopping container...'; docker rm -f $CONTAINER 2>/dev/null || true" EXIT

RESULT_DIR="/workspace/project/results/chiayi_v11_36e_65vids"
CONFIG="/workspace/project/configs/co_dino_chiayi_v11_36e_65vids.py"
DATA_ROOT="/workspace/datasets/coco/chiayi_v11_garbage_litter_corrected"
V8_RESULT_DIR="/workspace/hdd2_storage/chiayi_v10_36e_65vids"
V8_ERROR_DIR="${V8_RESULT_DIR}/error_analysis_score05"

CHECKPOINT=$(docker exec "$CONTAINER" bash -c \
    "ls ${RESULT_DIR}/best_coco_litter_precision_epoch_*.pth 2>/dev/null | tail -1")
echo "使用 best checkpoint: ${CHECKPOINT}"

echo "=== Part 2: AP 曲線分析 ==="
docker exec "$CONTAINER" python /workspace/project/post_analysis_9th.py \
    --mode ap-curve \
    --log "${RESULT_DIR}/training_log.md" \
    --output-dir "${RESULT_DIR}/error_analysis_score05"

echo "=== Part 3a: 場景分組分析 (score_thr=0.5) ==="
docker exec "$CONTAINER" python /workspace/project/post_analysis_9th.py \
    --mode by-scene \
    --error-dir "${RESULT_DIR}/error_analysis_score05" \
    --output-dir "${RESULT_DIR}/error_analysis_score05" \
    --output-name litter_garbage_by_scene_score05.md

echo "=== Part 3b: 場景分組分析 (score_thr=0.3) ==="
docker exec "$CONTAINER" python /workspace/project/post_analysis_9th.py \
    --mode by-scene \
    --error-dir "${RESULT_DIR}/error_analysis_score03" \
    --output-dir "${RESULT_DIR}/error_analysis_score03" \
    --output-name litter_garbage_by_scene_score03.md

echo "=== Part 4: Score 分布分析 ==="
docker exec -e CUDA_VISIBLE_DEVICES=0 "$CONTAINER" python /workspace/project/post_analysis_9th.py \
    --mode score-dist \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    --output-dir "${RESULT_DIR}/error_analysis_score05"

echo "=== Part 5: 第八次 vs 第九次對比 ==="
docker exec "$CONTAINER" python /workspace/project/post_analysis_9th.py \
    --mode comparison \
    --v7-log "${V8_RESULT_DIR}/training_log.md" \
    --v8-log "${RESULT_DIR}/training_log.md" \
    --v7-error-dir "$V8_ERROR_DIR" \
    --v8-error-dir "${RESULT_DIR}/error_analysis_score05" \
    --output-dir "${RESULT_DIR}/error_analysis_score05"

echo "=== Part 6: Litter/Garbage TP 視覺化比較（8th vs 9th）==="
V8_CHECKPOINT=$(docker exec "$CONTAINER" bash -c \
    "ls ${V8_RESULT_DIR}/best_coco_litter_precision_epoch_*.pth 2>/dev/null | tail -1")
if [ -z "$V8_CHECKPOINT" ]; then
    echo "WARNING: 找不到第八次 best checkpoint，跳過視覺化比較。"
else
    docker exec -e CUDA_VISIBLE_DEVICES=0 "$CONTAINER" \
      python /workspace/project/visualize_litter_tp_comparison_9th.py \
        --v7-checkpoint "$V8_CHECKPOINT" \
        --v8-checkpoint "$CHECKPOINT" \
        --v7-ea-dir     "${V8_RESULT_DIR}/error_analysis_score05" \
        --v8-ea-dir     "${RESULT_DIR}/error_analysis_score05" \
        --output-dir    "${RESULT_DIR}/litter_tp_comparison" \
      || echo "WARNING: Part 6 視覺化比較失敗，跳過（不影響後續流程）"
fi

echo ""
echo "=== 全部完成 ==="
