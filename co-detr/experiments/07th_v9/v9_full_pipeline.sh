#!/bin/bash
# v9 完整訓練 + 分析 Pipeline
#
# 流程：
#   Phase 1 — 監控 GPU 1（A6000），等空閒 >= 36 GB
#   Phase 2 — 啟動訓練，等待 36 epoch 完成
#   Phase 3 — Error Analysis（train / valid / test）
#   Phase 4 — Post Analysis（ap-curve / by-scene / score-dist / comparison）
#
# 使用方式：
#   nohup bash /path/to/repo/co-detr/v9_full_pipeline.sh > /dev/null 2>&1 &
#
# 查看進度：
#   tail -f /path/to/repo/co-detr/results/v9_pipeline.log

# ── 設定 ─────────────────────────────────────────────────────────────────────

CONTAINER="co-detr-test"
GPU_ID=1
REQUIRED_FREE_MB=30000
CHECK_INTERVAL=60

# container 內路徑
CONFIG="/workspace/project/configs/co_dino_chiayi_v9_36e.py"
DATA_ROOT="/workspace/datasets/coco/chiayi_v9_garbage_litter"
RESULTS_DIR="/workspace/project/results/chiayi_v9_36e"
EA_DIR="${RESULTS_DIR}/error_analysis"
TRAIN_LOG="${RESULTS_DIR}/autostart.log"
EA_LOG="${RESULTS_DIR}/error_analysis.log"
V9_TRAINING_LOG="${RESULTS_DIR}/training_log.md"
V8_TRAINING_LOG="/workspace/project/results/chiayi_v8_dense_36e/training_log.md"
V8_EA_DIR="/workspace/project/results/chiayi_v8_dense_36e/error_analysis"

# host 路徑
PIPELINE_LOG="/path/to/repo/co-detr/results/v9_pipeline.log"

# ── 工具函式 ──────────────────────────────────────────────────────────────────

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$PIPELINE_LOG"
}

gpu_free_mb() {
    nvidia-smi --id="$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
}

is_running_in_container() {
    local pattern="$1"
    docker exec "$CONTAINER" pgrep -f "$pattern" > /dev/null 2>&1
}

container_file_exists() {
    docker exec "$CONTAINER" test -f "$1" 2>/dev/null
}

# ── Phase 1: GPU 監控 ─────────────────────────────────────────────────────────

log "=========================================="
log "v9 Full Pipeline 啟動"
log "=========================================="
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

TRAIN_CMD="mkdir -p ${RESULTS_DIR} && \
    cd /opt/mmdetection && \
    PYTHONPATH=/opt/mmdetection:/workspace/project \
    CUDA_VISIBLE_DEVICES=${GPU_ID} \
    nohup python tools/train.py ${CONFIG} \
    --resume \
    --cfg-options train_dataloader.num_workers=4 train_dataloader.persistent_workers=True \
    > ${TRAIN_LOG} 2>&1"

docker exec -d "$CONTAINER" bash -c "$TRAIN_CMD"

sleep 15

if ! is_running_in_container "tools/train.py"; then
    log "ERROR: 訓練啟動失敗！請檢查 ${TRAIN_LOG}"
    exit 1
fi

log "  訓練已啟動。"
log "  訓練 log：${TRAIN_LOG}"
log ""
log "  等待 36 epoch 完成..."

while is_running_in_container "tools/train.py"; do
    # 從 training_log.md 取最新 epoch 進度
    LATEST_EP=$(docker exec "$CONTAINER" bash -c \
        "grep '^## Epoch' ${V9_TRAINING_LOG} 2>/dev/null | tail -1" 2>/dev/null)
    log "  訓練中... ${LATEST_EP:-（尚無 log）}"
    sleep 300
done

log "  訓練進程結束。"

# 確認 best checkpoint
BEST_CKPT=$(docker exec "$CONTAINER" bash -c \
    "ls ${RESULTS_DIR}/best_coco_bbox_mAP_epoch_*.pth 2>/dev/null | head -1")

if [ -z "$BEST_CKPT" ]; then
    log "ERROR: 找不到 best checkpoint，請確認訓練是否正常完成。"
    exit 1
fi

log "  Best checkpoint: ${BEST_CKPT}"

# ── Phase 3: Error Analysis ───────────────────────────────────────────────────

log ""
log "Phase 3: Error Analysis（train / valid / test）"

docker exec -d "$CONTAINER" bash -c "\
    PYTHONPATH=/opt/mmdetection:/workspace/project \
    python /workspace/project/error_analysis_v9.py \
        --config ${CONFIG} \
        --checkpoint ${BEST_CKPT} \
        --data-root ${DATA_ROOT} \
        --output-dir ${EA_DIR} \
        --splits train valid test \
    > ${EA_LOG} 2>&1"

sleep 10
log "  Error Analysis 已啟動，log：${EA_LOG}"

# 等待完成（同時確認 process 還活著）
while ! container_file_exists "${EA_DIR}/error_analysis_report.md"; do
    if ! is_running_in_container "error_analysis_v9"; then
        log "ERROR: error_analysis_v9 進程意外結束！請檢查 ${EA_LOG}"
        exit 1
    fi
    log "  Error Analysis 執行中..."
    sleep 300
done

log "  Error Analysis 完成！"

# ── Phase 4: Post Analysis ────────────────────────────────────────────────────

log ""
log "Phase 4: Post Analysis"

run_post() {
    local mode="$1"
    local extra_args="$2"
    log "  [${mode}] 開始..."
    docker exec "$CONTAINER" bash -c "\
        PYTHONPATH=/opt/mmdetection:/workspace/project \
        python /workspace/project/post_analysis_v9.py \
            --mode ${mode} \
            --output-dir ${EA_DIR} \
            ${extra_args} \
        >> ${EA_LOG} 2>&1"
    if [ $? -eq 0 ]; then
        log "  [${mode}] 完成。"
    else
        log "  [${mode}] 執行失敗，請檢查 ${EA_LOG}"
    fi
}

run_post "ap-curve" \
    "--v9-log ${V9_TRAINING_LOG}"

run_post "by-scene" \
    "--error-dir ${EA_DIR}"

run_post "score-dist" \
    "--config ${CONFIG} --checkpoint ${BEST_CKPT} --data-root ${DATA_ROOT}"

run_post "comparison" \
    "--v8-log ${V8_TRAINING_LOG} --v9-log ${V9_TRAINING_LOG} \
     --v8-error-dir ${V8_EA_DIR} --v9-error-dir ${EA_DIR}"

# ── 完成 ─────────────────────────────────────────────────────────────────────

log ""
log "=========================================="
log "Pipeline 全部完成！"
log ""
log "輸出目錄：${EA_DIR}"
log ""
log "產出檔案："
log "  error_analysis_report.md"
log "  {train,valid,test}_per_frame_errors.csv"
log "  {train,valid,test}_per_frame_class_errors.csv"
log "  {train,valid,test}_confusion_matrix.csv"
log "  litter_garbage_ap_curve.md"
log "  litter_garbage_by_scene.md"
log "  score_distribution.md"
log "  v9_vs_v8_comparison.md"
log "  litter_garbage_fp_visualizations/"
log "=========================================="
