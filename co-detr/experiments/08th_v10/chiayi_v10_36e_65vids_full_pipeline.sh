#!/bin/bash
# 第八次實驗完整 Pipeline：chiayi_v10_36e_65vids
#
# 流程：
#   Phase 1 — 監控 GPU 1（A6000），等空閒 >= 30 GB
#   Phase 2 — 啟動訓練，等待 36 epoch 完成
#   Phase 3 — Error Analysis（score_thr=0.5 和 0.3，三個 split）
#   Phase 4 — Post Analysis（AP曲線 / 場景分析 / score分布 / 7th vs 8th 對比）
#   Phase 5 — Litter/Garbage TP 視覺化比較（7th vs 8th，三個 split）
#
# 使用方式（背景執行，全程 log 到 pipeline.log）：
#   nohup bash /path/to/repo/co-detr/chiayi_v10_36e_65vids_full_pipeline.sh > /dev/null 2>&1 &
#
# 查看進度：
#   tail -f /path/to/repo/co-detr/results/chiayi_v10_36e_65vids/pipeline.log

set -euo pipefail

# ── 設定 ─────────────────────────────────────────────────────────────────────

CONTAINER="co-detr-test"
GPU_ID=1
REQUIRED_FREE_MB=30000
CHECK_INTERVAL=60

# container 內路徑
CONFIG="/workspace/project/configs/co_dino_chiayi_v10_36e_65vids.py"
DATA_ROOT="/workspace/datasets/coco/chiayi_v10_garbage_litter"
RESULT_DIR="/workspace/project/results/chiayi_v10_36e_65vids"
TRAIN_LOG="${RESULT_DIR}/train.log"
TRAINING_MD="${RESULT_DIR}/training_log.md"

V7_RESULT_DIR="/workspace/project/results/7th_expt_v9_loss_compensate_36e"
V7_ERROR_DIR="${V7_RESULT_DIR}/error_analysis_score05"

# host 路徑（log）
PIPELINE_LOG="/path/to/repo/co-detr/results/chiayi_v10_36e_65vids/pipeline.log"

# ── 工具函式 ──────────────────────────────────────────────────────────────────

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$PIPELINE_LOG"
}

gpu_free_mb() {
    nvidia-smi --id="$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
}

is_training() {
    docker exec "$CONTAINER" pgrep -f "tools/train.py" > /dev/null 2>&1
}

container_file_exists() {
    docker exec "$CONTAINER" test -f "$1" 2>/dev/null
}

# ── 初始化 ────────────────────────────────────────────────────────────────────

mkdir -p "$(dirname "$PIPELINE_LOG")"
docker exec "$CONTAINER" bash -c "chmod 777 ${RESULT_DIR} 2>/dev/null || true"

log "=========================================="
log "chiayi_v10_36e_65vids Full Pipeline 啟動"
log "=========================================="

# ── Phase 1: GPU 監控 ─────────────────────────────────────────────────────────

log ""
log "Phase 1: 監控 GPU ${GPU_ID}（需要空閒 >= ${REQUIRED_FREE_MB} MB）"

while true; do
    FREE=$(gpu_free_mb)
    log "  GPU ${GPU_ID} 空閒: ${FREE} MB"
    if [ "$FREE" -ge "$REQUIRED_FREE_MB" ]; then
        log "  GPU 空閒足夠，準備啟動訓練。"
        break
    fi
    sleep "$CHECK_INTERVAL"
done

# ── Phase 2: 啟動訓練 + 等待完成 ─────────────────────────────────────────────

log ""
log "Phase 2: 啟動訓練"

TRAIN_CMD="cd /opt/mmdetection && \
    PYTHONPATH=/opt/mmdetection:/workspace/project \
    nohup python tools/train.py ${CONFIG} \
    --resume \
    > ${TRAIN_LOG} 2>&1"

docker exec -d "$CONTAINER" bash -c "$TRAIN_CMD"
sleep 15

if ! is_training; then
    log "ERROR: 訓練啟動失敗！請檢查 ${TRAIN_LOG}"
    exit 1
fi

log "  訓練已啟動。log：${TRAIN_LOG}"
log "  等待 36 epoch 完成（每 5 分鐘報告進度）..."

while is_training; do
    LATEST_EP=$(docker exec "$CONTAINER" bash -c \
        "grep '^## Epoch' ${TRAINING_MD} 2>/dev/null | tail -1" 2>/dev/null || true)
    log "  訓練中... ${LATEST_EP:-（尚無 epoch log）}"
    sleep 300
done

log "  訓練進程結束。"

# 確認 best checkpoint（save_best=litter_precision）
CHECKPOINT=$(docker exec "$CONTAINER" bash -c \
    "ls ${RESULT_DIR}/best_coco_litter_precision_epoch_*.pth 2>/dev/null | tail -1")

if [ -z "$CHECKPOINT" ]; then
    log "ERROR: 找不到 best_coco_litter_precision_epoch_*.pth，請確認訓練是否正常完成。"
    exit 1
fi
log "  Best checkpoint: ${CHECKPOINT}"

# 確認第七次 best checkpoint（視覺化用）
V7_CHECKPOINT=$(docker exec "$CONTAINER" bash -c \
    "ls ${V7_RESULT_DIR}/best_coco_litter_precision_epoch_*.pth 2>/dev/null | tail -1")

# ── Phase 3a: Error Analysis（score_thr=0.5）────────────────────────────────

log ""
log "Phase 3a: Error Analysis（score_thr=0.5，train/valid/test）"

docker exec -e CUDA_VISIBLE_DEVICES=1 "$CONTAINER" \
    python /workspace/project/error_analysis_v9.py \
        --config      "$CONFIG" \
        --checkpoint  "$CHECKPOINT" \
        --data-root   "$DATA_ROOT" \
        --output-dir  "${RESULT_DIR}/error_analysis_score05" \
        --splits train valid test \
        --score-threshold 0.5 \
    2>&1 | tee -a "$PIPELINE_LOG"

log "  Error Analysis (0.5) 完成。"

# ── Phase 3b: Error Analysis（score_thr=0.3）────────────────────────────────

log ""
log "Phase 3b: Error Analysis（score_thr=0.3，train/valid/test）"

docker exec -e CUDA_VISIBLE_DEVICES=1 "$CONTAINER" \
    python /workspace/project/error_analysis_v9.py \
        --config      "$CONFIG" \
        --checkpoint  "$CHECKPOINT" \
        --data-root   "$DATA_ROOT" \
        --output-dir  "${RESULT_DIR}/error_analysis_score03" \
        --splits train valid test \
        --score-threshold 0.3 \
    2>&1 | tee -a "$PIPELINE_LOG"

log "  Error Analysis (0.3) 完成。"

# ── Phase 4a: AP 曲線分析 ─────────────────────────────────────────────────────

log ""
log "Phase 4a: AP 曲線分析"

docker exec "$CONTAINER" \
    python /workspace/project/post_analysis_8th.py \
        --mode ap-curve \
        --log        "${RESULT_DIR}/training_log.md" \
        --output-dir "${RESULT_DIR}/error_analysis_score05" \
    2>&1 | tee -a "$PIPELINE_LOG"

# ── Phase 4b: 場景分組分析 ───────────────────────────────────────────────────

log ""
log "Phase 4b: 場景分組分析（score_thr=0.5）"

docker exec "$CONTAINER" \
    python /workspace/project/post_analysis_8th.py \
        --mode by-scene \
        --error-dir  "${RESULT_DIR}/error_analysis_score05" \
        --output-dir "${RESULT_DIR}/error_analysis_score05" \
        --output-name litter_garbage_by_scene_score05.md \
    2>&1 | tee -a "$PIPELINE_LOG"

log "Phase 4c: 場景分組分析（score_thr=0.3）"

docker exec "$CONTAINER" \
    python /workspace/project/post_analysis_8th.py \
        --mode by-scene \
        --error-dir  "${RESULT_DIR}/error_analysis_score03" \
        --output-dir "${RESULT_DIR}/error_analysis_score03" \
        --output-name litter_garbage_by_scene_score03.md \
    2>&1 | tee -a "$PIPELINE_LOG"

# ── Phase 4d: Confidence Score 分布 ──────────────────────────────────────────

log ""
log "Phase 4d: Confidence Score 分布（test set 全量推論）"

docker exec -e CUDA_VISIBLE_DEVICES=1 "$CONTAINER" \
    python /workspace/project/post_analysis_8th.py \
        --mode score-dist \
        --config     "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --data-root  "$DATA_ROOT" \
        --output-dir "${RESULT_DIR}/error_analysis_score05" \
    2>&1 | tee -a "$PIPELINE_LOG"

# ── Phase 4e: 第七次 vs 第八次對比報告 ───────────────────────────────────────

log ""
log "Phase 4e: 第七次 vs 第八次對比報告"

docker exec "$CONTAINER" \
    python /workspace/project/post_analysis_8th.py \
        --mode comparison \
        --v7-log       "${V7_RESULT_DIR}/training_log.md" \
        --v8-log       "${RESULT_DIR}/training_log.md" \
        --v7-error-dir "$V7_ERROR_DIR" \
        --v8-error-dir "${RESULT_DIR}/error_analysis_score05" \
        --output-dir   "${RESULT_DIR}/error_analysis_score05" \
    2>&1 | tee -a "$PIPELINE_LOG"

# ── Phase 5: Litter/Garbage TP 視覺化比較 ────────────────────────────────────

log ""
log "Phase 5: Litter/Garbage TP 視覺化比較（7th vs 8th，三個 split）"

if [ -z "$V7_CHECKPOINT" ]; then
    log "  WARNING: 找不到第七次 best checkpoint，跳過視覺化比較。"
else
    docker exec -e CUDA_VISIBLE_DEVICES=1 "$CONTAINER" \
        python /workspace/project/visualize_litter_tp_comparison_8th.py \
            --v7-checkpoint "$V7_CHECKPOINT" \
            --v8-checkpoint "$CHECKPOINT" \
            --v7-ea-dir     "$V7_ERROR_DIR" \
            --v8-ea-dir     "${RESULT_DIR}/error_analysis_score05" \
            --output-dir    "${RESULT_DIR}/litter_tp_comparison" \
        2>&1 | tee -a "$PIPELINE_LOG"
fi

# ── 完成 ─────────────────────────────────────────────────────────────────────

log ""
log "=========================================="
log "Pipeline 全部完成！"
log ""
log "主要輸出："
log "  AP 曲線:          ${RESULT_DIR}/error_analysis_score05/litter_garbage_ap_curve.md"
log "  場景分析(0.5):    ${RESULT_DIR}/error_analysis_score05/litter_garbage_by_scene_score05.md"
log "  場景分析(0.3):    ${RESULT_DIR}/error_analysis_score03/litter_garbage_by_scene_score03.md"
log "  Score 分布:       ${RESULT_DIR}/error_analysis_score05/score_distribution.md"
log "  7th vs 8th 對比:  ${RESULT_DIR}/error_analysis_score05/v10_7th_vs_8th_comparison.md"
log "  TP 視覺化(7th):   ${RESULT_DIR}/litter_tp_comparison/7th/{train,valid,test}/"
log "  TP 視覺化(8th):   ${RESULT_DIR}/litter_tp_comparison/8th/{train,valid,test}/"
log "=========================================="
