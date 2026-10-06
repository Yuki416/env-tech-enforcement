#!/usr/bin/env bash
# 第八次實驗訓練後全套分析腳本（chiayi_v10_36e_65vids）
#
# 使用方式：
#   bash chiayi_v10_36e_65vids_post_analysis.sh
#
# 前提：訓練已完成，best_coco_litter_precision_epoch_*.pth 已存在。

set -euo pipefail

CONTAINER="co-detr-test"
RESULT_DIR="/workspace/hdd2_storage/chiayi_v10_36e_65vids"
CONFIG="/workspace/hdd2_storage/chiayi_v10_36e_65vids/co_dino_chiayi_v10_36e_65vids.py"
DATA_ROOT="/workspace/hdd2_storage/datasets/coco/chiayi_v10_garbage_litter"

CHECKPOINT=$(docker exec "$CONTAINER" bash -c \
    "ls ${RESULT_DIR}/best_coco_litter_precision_epoch_*.pth 2>/dev/null | tail -1")
if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: 找不到 best_coco_litter_precision_epoch_*.pth，請確認訓練已完成。"
    exit 1
fi
echo "使用 best checkpoint: ${CHECKPOINT}"

V7_RESULT_DIR="/workspace/hdd2_storage/7th_expt_v9_loss_compensate_36e"
V7_ERROR_DIR="${V7_RESULT_DIR}/error_analysis_score05"

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
docker exec "$CONTAINER" python /workspace/project/post_analysis_8th.py \
    --mode ap-curve \
    --log "${RESULT_DIR}/training_log.md" \
    --output-dir "${RESULT_DIR}/error_analysis_score05"

# ─────────────────────────────────────────────────────────────────────────────
# Part 3a: 場景分組分析（score_thr=0.5）
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Part 3a: 場景分組分析 (score_thr=0.5) ==="
docker exec "$CONTAINER" python /workspace/project/post_analysis_8th.py \
    --mode by-scene \
    --error-dir "${RESULT_DIR}/error_analysis_score05" \
    --output-dir "${RESULT_DIR}/error_analysis_score05" \
    --output-name litter_garbage_by_scene_score05.md

# ─────────────────────────────────────────────────────────────────────────────
# Part 3b: 場景分組分析（score_thr=0.3）
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Part 3b: 場景分組分析 (score_thr=0.3) ==="
docker exec "$CONTAINER" python /workspace/project/post_analysis_8th.py \
    --mode by-scene \
    --error-dir "${RESULT_DIR}/error_analysis_score03" \
    --output-dir "${RESULT_DIR}/error_analysis_score03" \
    --output-name litter_garbage_by_scene_score03.md

# ─────────────────────────────────────────────────────────────────────────────
# Part 4: Confidence score 分布（低閾值全量推論）
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Part 4: Score 分布分析 ==="
docker exec -e CUDA_VISIBLE_DEVICES=1 "$CONTAINER" python /workspace/project/post_analysis_8th.py \
    --mode score-dist \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --data-root "$DATA_ROOT" \
    --output-dir "${RESULT_DIR}/error_analysis_score05"

# ─────────────────────────────────────────────────────────────────────────────
# Part 5: 第七次 vs 第八次對比
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Part 5: 第七次 vs 第八次對比 ==="
docker exec "$CONTAINER" python /workspace/project/post_analysis_8th.py \
    --mode comparison \
    --v7-log "${V7_RESULT_DIR}/training_log.md" \
    --v8-log "${RESULT_DIR}/training_log.md" \
    --v7-error-dir "$V7_ERROR_DIR" \
    --v8-error-dir "${RESULT_DIR}/error_analysis_score05" \
    --output-dir "${RESULT_DIR}/error_analysis_score05"

# ─────────────────────────────────────────────────────────────────────────────
# Part 6: Litter/Garbage TP frame 視覺化比較（7th vs 8th，三個 split）
# ─────────────────────────────────────────────────────────────────────────────
echo "=== Part 6: Litter/Garbage TP 視覺化比較（7th vs 8th）==="

V7_CHECKPOINT=$(docker exec "$CONTAINER" bash -c \
    "ls ${V7_RESULT_DIR}/best_coco_litter_precision_epoch_*.pth 2>/dev/null | tail -1")
if [ -z "$V7_CHECKPOINT" ]; then
    echo "WARNING: 找不到第七次 best checkpoint，跳過視覺化比較。"
else
    docker exec -e CUDA_VISIBLE_DEVICES=1 "$CONTAINER" \
      python /workspace/project/visualize_litter_tp_comparison_8th.py \
        --v7-checkpoint "$V7_CHECKPOINT" \
        --v8-checkpoint "$CHECKPOINT" \
        --v7-ea-dir     "${V7_RESULT_DIR}/error_analysis_score05" \
        --v8-ea-dir     "${RESULT_DIR}/error_analysis_score05" \
        --output-dir    "${RESULT_DIR}/litter_tp_comparison"
fi

echo ""
echo "=== 全部完成 ==="
echo "輸出目錄："
echo "  score_thr=0.5: ${RESULT_DIR}/error_analysis_score05/"
echo "  score_thr=0.3: ${RESULT_DIR}/error_analysis_score03/"
echo "  TP 視覺化:     ${RESULT_DIR}/litter_tp_comparison/"
echo ""
echo "主要報告："
echo "  AP 曲線:          ${RESULT_DIR}/error_analysis_score05/litter_garbage_ap_curve.md"
echo "  場景分析(0.5):    ${RESULT_DIR}/error_analysis_score05/litter_garbage_by_scene_score05.md"
echo "  場景分析(0.3):    ${RESULT_DIR}/error_analysis_score03/litter_garbage_by_scene_score03.md"
echo "  Score 分布:       ${RESULT_DIR}/error_analysis_score05/score_distribution.md"
echo "  7th vs 8th 對比:  ${RESULT_DIR}/error_analysis_score05/v10_7th_vs_8th_comparison.md"
echo "  TP 視覺化(7th):   ${RESULT_DIR}/litter_tp_comparison/7th/{train,valid,test}/"
echo "  TP 視覺化(8th):   ${RESULT_DIR}/litter_tp_comparison/8th/{train,valid,test}/"
