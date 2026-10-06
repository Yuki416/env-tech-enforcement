#!/usr/bin/env bash
# 第七次實驗訓練後全套分析腳本
#
# 使用方式：
#   bash 7th_expt_post_analysis.sh
#
# 前提：訓練已完成，epoch_36.pth 已存在。

set -euo pipefail

CONTAINER="co-detr-test"
RESULT_DIR="/workspace/project/results/7th_expt_v9_loss_compensate_36e"
CONFIG="/workspace/project/configs/co_dino_7th_expt_v9_loss_compensate_36e.py"
DATA_ROOT="/workspace/datasets/coco/chiayi_v9_garbage_litter"

# 使用 best checkpoint（litter_precision 最高點，ep5）
CHECKPOINT=$(docker exec "$CONTAINER" bash -c \
    "ls ${RESULT_DIR}/best_coco_litter_precision_epoch_*.pth 2>/dev/null | tail -1")
if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: 找不到 best_coco_litter_precision_epoch_*.pth，請確認訓練已完成。"
    exit 1
fi
echo "使用 best checkpoint: ${CHECKPOINT}"

V6_RESULT_DIR="/workspace/project/results/chiayi_v9_36e"
V6_ERROR_DIR="${V6_RESULT_DIR}/error_analysis"

# ─────────────────────────────────────────────────────────────────────────────
# Part 1a: Error analysis — score_thr=0.5
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Part 1a: Error analysis (score_thr=0.5) ==="
docker exec -e CUDA_VISIBLE_DEVICES=1 "$CONTAINER" python /workspace/project/error_analysis_v9.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    --output-dir "${RESULT_DIR}/error_analysis_score05" \
    --splits train valid test \
    --score-threshold 0.5

# ─────────────────────────────────────────────────────────────────────────────
# Part 1b: Error analysis — score_thr=0.3
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Part 1b: Error analysis (score_thr=0.3) ==="
docker exec -e CUDA_VISIBLE_DEVICES=1 "$CONTAINER" python /workspace/project/error_analysis_v9.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    --output-dir "${RESULT_DIR}/error_analysis_score03" \
    --splits train valid test \
    --score-threshold 0.3

# ─────────────────────────────────────────────────────────────────────────────
# Part 2: AP 曲線分析
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Part 2: AP 曲線分析 ==="
docker exec "$CONTAINER" python /workspace/project/post_analysis_7th.py \
    --mode ap-curve \
    --log "${RESULT_DIR}/training_log.md" \
    --output-dir "${RESULT_DIR}/error_analysis_score05"

# ─────────────────────────────────────────────────────────────────────────────
# Part 3a: 場景分組分析（score_thr=0.5）
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Part 3a: 場景分組分析 (score_thr=0.5) ==="
docker exec "$CONTAINER" python /workspace/project/post_analysis_7th.py \
    --mode by-scene \
    --error-dir "${RESULT_DIR}/error_analysis_score05" \
    --output-dir "${RESULT_DIR}/error_analysis_score05" \
    --output-name litter_garbage_by_scene_score05.md

# ─────────────────────────────────────────────────────────────────────────────
# Part 3b: 場景分組分析（score_thr=0.3）
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Part 3b: 場景分組分析 (score_thr=0.3) ==="
docker exec "$CONTAINER" python /workspace/project/post_analysis_7th.py \
    --mode by-scene \
    --error-dir "${RESULT_DIR}/error_analysis_score03" \
    --output-dir "${RESULT_DIR}/error_analysis_score03" \
    --output-name litter_garbage_by_scene_score03.md

# ─────────────────────────────────────────────────────────────────────────────
# Part 4: Confidence score 分布（低閾值全量推論）
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Part 4: Score 分布分析 ==="
docker exec -e CUDA_VISIBLE_DEVICES=1 "$CONTAINER" python /workspace/project/post_analysis_7th.py \
    --mode score-dist \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    --output-dir "${RESULT_DIR}/error_analysis_score05"

# ─────────────────────────────────────────────────────────────────────────────
# Part 5: 第六次 vs 第七次對比
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Part 5: 第六次 vs 第七次對比 ==="
docker exec "$CONTAINER" python /workspace/project/post_analysis_7th.py \
    --mode comparison \
    --v6-log "${V6_RESULT_DIR}/training_log.md" \
    --v7-log "${RESULT_DIR}/training_log.md" \
    --v6-error-dir "$V6_ERROR_DIR" \
    --v7-error-dir "${RESULT_DIR}/error_analysis_score05" \
    --output-dir "${RESULT_DIR}/error_analysis_score05"

echo ""
echo "=== 全部完成 ==="
echo "輸出目錄："
echo "  score_thr=0.5: ${RESULT_DIR}/error_analysis_score05/"
echo "  score_thr=0.3: ${RESULT_DIR}/error_analysis_score03/"
echo ""
echo "主要報告："
echo "  AP 曲線:          ${RESULT_DIR}/error_analysis_score05/litter_garbage_ap_curve.md"
echo "  場景分析(0.5):    ${RESULT_DIR}/error_analysis_score05/litter_garbage_by_scene_score05.md"
echo "  場景分析(0.3):    ${RESULT_DIR}/error_analysis_score03/litter_garbage_by_scene_score03.md"
echo "  Score 分布:       ${RESULT_DIR}/error_analysis_score05/score_distribution.md"
echo "  6th vs 7th 對比:  ${RESULT_DIR}/error_analysis_score05/v9_6th_vs_7th_comparison.md"
