#!/usr/bin/env bash
# wait_and_train_v11.sh
#
# 自動流程：
#   Phase 1 — 等待 v11 zip 下載完成
#   Phase 2 — 解壓縮，自動偵測資料夾名稱，不符時建立 symlink
#   Phase 3 — 等待雙 GPU 空閒（每顆 >= 18 GB）
#   Phase 4 — 啟動 Co-DETR v11 DDP 訓練
#
# 使用方式（背景執行）：
#   nohup bash wait_and_train_v11.sh > /dev/null 2>&1 &
#
# 查看進度：
#   tail -f results/chiayi_v11_36e_65vids/wait_and_train.log

set -uo pipefail

# ── 設定 ─────────────────────────────────────────────────────────────────────
ZIP_PATH="/path/to/data/coco/chiayi_v11_garbage_litter_corrected_65vids.zip"
DATASET_DIR="/path/to/data/coco"
EXPECTED_FOLDER="chiayi_v11_garbage_litter_corrected"

GPU_FREE_THRESHOLD=18000   # MiB，18 GB（Co-DETR bs=1 約需 14-16 GB）
CHECK_INTERVAL=60          # 秒

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${SCRIPT_DIR}/results/chiayi_v11_36e_65vids/wait_and_train.log"
TRAIN_LOG="${SCRIPT_DIR}/results/chiayi_v11_36e_65vids/train.log"
CONFIG="/workspace/project/configs/co_dino_chiayi_v11_36e_65vids.py"

# ── 工具函式 ──────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

gpu_free_mib() {
    # $1 = 0-indexed GPU id
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
        | awk "NR==$(( $1 + 1 ))"
}

# ── 初始化 ────────────────────────────────────────────────────────────────────
mkdir -p "$(dirname "$LOG")"

log "========================================================"
log " Co-DETR v11 自動等待並啟動訓練"
log "========================================================"

# ── Phase 1: 等待 zip 下載完成 ────────────────────────────────────────────────
log ""
log "Phase 1: 等待 zip 下載完成"
log "  目標: $ZIP_PATH"

while true; do
    PARTIAL=$(ls "${ZIP_PATH}".*.partial 2>/dev/null | head -1 || true)
    if [ -n "$PARTIAL" ]; then
        DONE_SIZE=$(du -h "$PARTIAL" | cut -f1)
        TOTAL_SIZE=$(du -h "$ZIP_PATH" 2>/dev/null | cut -f1 || echo "?")
        log "  下載中... 已下載 ${DONE_SIZE}"
    elif [ -f "$ZIP_PATH" ]; then
        ZIP_SIZE=$(du -h "$ZIP_PATH" | cut -f1)
        log "  下載完成！大小: $ZIP_SIZE"
        break
    else
        log "  等待 zip 出現..."
    fi
    sleep "$CHECK_INTERVAL"
done

# ── Phase 2: 解壓縮 ───────────────────────────────────────────────────────────
log ""
log "Phase 2: 解壓縮"

if [ -d "${DATASET_DIR}/${EXPECTED_FOLDER}" ]; then
    log "  目錄已存在，跳過解壓縮: ${DATASET_DIR}/${EXPECTED_FOLDER}"
else
    # 偵測 zip 內的根目錄名稱
    ACTUAL_FOLDER=$(unzip -Z1 "$ZIP_PATH" 2>/dev/null | head -1 | cut -d/ -f1)
    log "  zip 內根目錄: ${ACTUAL_FOLDER}"
    log "  開始解壓縮至 $DATASET_DIR ..."

    unzip -O CP950 -q "$ZIP_PATH" -d "$DATASET_DIR" 2>&1 | tee -a "$LOG" || \
    unzip -q "$ZIP_PATH" -d "$DATASET_DIR" 2>&1 | tee -a "$LOG"

    log "  解壓完成"

    # 名稱不符時建立 symlink
    if [ -n "$ACTUAL_FOLDER" ] && [ "$ACTUAL_FOLDER" != "$EXPECTED_FOLDER" ]; then
        if [ -d "${DATASET_DIR}/${ACTUAL_FOLDER}" ]; then
            log "  資料夾名稱與 config 不符，建立 symlink:"
            log "    ${DATASET_DIR}/${EXPECTED_FOLDER} -> ${ACTUAL_FOLDER}"
            ln -sfn "${DATASET_DIR}/${ACTUAL_FOLDER}" "${DATASET_DIR}/${EXPECTED_FOLDER}"
        else
            log "  ERROR: 找不到解壓後的目錄 ${DATASET_DIR}/${ACTUAL_FOLDER}"
            exit 1
        fi
    fi
fi

# 驗證資料集結構
log "  驗證資料集結構..."
DATASET_OK=true
for SPLIT in train valid test; do
    IMG_COUNT=$(ls "${DATASET_DIR}/${EXPECTED_FOLDER}/${SPLIT}/" 2>/dev/null | wc -l || echo 0)
    ANN_FILE="${DATASET_DIR}/${EXPECTED_FOLDER}/annotations/${SPLIT}.json"
    if [ -f "$ANN_FILE" ]; then
        ANN_STATUS="✓"
    else
        ANN_STATUS="✗ 缺少!"
        DATASET_OK=false
    fi
    log "    ${SPLIT}: ${IMG_COUNT} 張圖  |  annotations/${SPLIT}.json ${ANN_STATUS}"
done

if [ "$DATASET_OK" = false ]; then
    log "  ERROR: 資料集結構有問題，終止。"
    exit 1
fi
log "  資料集驗證通過 ✓"

# ── Phase 3: 等待雙 GPU 空閒 ──────────────────────────────────────────────────
log ""
log "Phase 3: 等待雙 GPU 空閒（閾值: ${GPU_FREE_THRESHOLD} MiB / GPU）"

while true; do
    GPU0=$(gpu_free_mib 0)
    GPU1=$(gpu_free_mib 1)
    log "  GPU 0 空閒: ${GPU0} MiB  |  GPU 1 空閒: ${GPU1} MiB  (需要 >= ${GPU_FREE_THRESHOLD})"

    if [ "${GPU0}" -ge "${GPU_FREE_THRESHOLD}" ] && \
       [ "${GPU1}" -ge "${GPU_FREE_THRESHOLD}" ]; then
        log "  雙 GPU 空閒足夠，準備啟動！"
        break
    fi
    sleep "$CHECK_INTERVAL"
done

# ── Phase 4: 啟動訓練 ─────────────────────────────────────────────────────────
log ""
log "Phase 4: 啟動 Co-DETR v11 DDP 訓練"
log "  Config: $CONFIG"
log "  Train log: $TRAIN_LOG"

nohup docker run --rm --gpus all \
    --ipc=host --shm-size=16g \
    -v "${SCRIPT_DIR}:/workspace/project" \
    -v /path/to/data/coco:/workspace/datasets/coco \
    -v /path/to/data:/workspace/hdd2_storage \
    -e PYTHONPATH=/opt/mmdetection:/workspace/project \
    -e LANG=C.UTF-8 \
    -e LC_ALL=C.UTF-8 \
    co-detr:latest \
    bash /opt/mmdetection/tools/dist_train.sh \
        "$CONFIG" \
        2 \
        --cfg-options train_dataloader.batch_size=1 val_dataloader.batch_size=1 \
    >> "$TRAIN_LOG" 2>&1 &

TRAIN_PID=$!
log "  訓練已啟動，Docker PID: $TRAIN_PID"
log ""
log "查看訓練進度:"
log "  tail -f $TRAIN_LOG"
log ""
log "========================================================"
log " 監控腳本結束，訓練在背景執行中"
log "========================================================"
