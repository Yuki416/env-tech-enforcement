#!/usr/bin/env bash
# wait_and_train_10th.sh
# 第 10 次訓練：等雙卡空閒後自動啟動 Co-DETR DDP 訓練（v12 litter-only）
# 資料集已預先建好（chiayi_v12_other_team_litter_annotation），故只含兩階段：
#   Phase 1 — 等待雙 GPU 空閒（每卡 >= 18 GB）
#   Phase 2 — 啟動雙卡 DDP 訓練
#
# 背景執行：
#   nohup bash wait_and_train_10th.sh > /dev/null 2>&1 &
# 查看進度：
#   tail -f /path/to/data/10th_expt_v12_other_team_litter_36e/wait_and_train.log
#   tail -f /path/to/data/10th_expt_v12_other_team_litter_36e/train.log

set -uo pipefail

# ── 設定 ─────────────────────────────────────────────────────────────────────
GPU_FREE_THRESHOLD=18000   # MiB，18 GB（Co-DETR bs=1 約需 14-16 GB，留緩衝）
CHECK_INTERVAL=60          # 秒

CONFIG="/workspace/project/configs/co_dino_chiayi_v12_litter_only_36e.py"
RESULT_DIR="/path/to/data/10th_expt_v12_other_team_litter_36e"
LOG="${RESULT_DIR}/wait_and_train.log"
TRAIN_LOG="${RESULT_DIR}/train.log"

mkdir -p "$RESULT_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

gpu_free_mib() {
    # $1 = 0-indexed GPU id
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | awk "NR==$(( $1 + 1 ))"
}

log "===================================================="
log " 第 10 次訓練 自動等待雙卡空閒並啟動（v12 litter-only）"
log " 閾值: 每卡 >= ${GPU_FREE_THRESHOLD} MiB"
log "===================================================="

# ── Phase 1: 等待雙 GPU 空閒 ──────────────────────────────────────────────────
log ""
log "Phase 1: 等待雙 GPU 空閒"
while true; do
    GPU0=$(gpu_free_mib 0)
    GPU1=$(gpu_free_mib 1)
    log "  GPU0 空閒: ${GPU0} MiB | GPU1 空閒: ${GPU1} MiB (需 >= ${GPU_FREE_THRESHOLD})"
    if [ "${GPU0}" -ge "${GPU_FREE_THRESHOLD}" ] && [ "${GPU1}" -ge "${GPU_FREE_THRESHOLD}" ]; then
        log "  雙卡空閒足夠，準備啟動！"
        break
    fi
    sleep "$CHECK_INTERVAL"
done

# ── Phase 2: 啟動 DDP 訓練（前景執行，blocking 直到訓練結束）──────────────────
log ""
log "Phase 2: 啟動 Co-DETR 第 10 次 DDP 訓練"
log "  Config: $CONFIG"
log "  Train log: $TRAIN_LOG"

docker run --rm --gpus all \
    --ipc=host --shm-size=16g \
    -v /path/to/repo/co-detr:/workspace/project \
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
    >> "$TRAIN_LOG" 2>&1

log ""
log "訓練程序結束（docker 已退出）。請檢查 $TRAIN_LOG 確認是否正常完成。"
