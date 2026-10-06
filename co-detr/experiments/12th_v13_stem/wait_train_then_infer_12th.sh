#!/usr/bin/env bash
# 等第 12 次訓練容器結束 → 自動啟動 v13 4ch video inference（解凍 stem 版）。
# 用法： nohup bash wait_train_then_infer_12th.sh <TRAIN_CID> > .../wait_then_infer.log 2>&1 &

set -uo pipefail

TRAIN_CID="${1:?需傳入第 12 次訓練容器 ID/名稱}"
PROJECT_DIR="/path/to/repo/co-detr"
OUT_DIR="/path/to/data/12th_expt_v13_bgs_4ch_stem_unfreeze_36e"

echo "[$(date '+%F %T')] 等待訓練容器 ${TRAIN_CID} 結束..."
docker wait "${TRAIN_CID}" >/dev/null 2>&1 || true
echo "[$(date '+%F %T')] 訓練容器已結束。"

# 確認 best ckpt 已存在
BEST=$(ls -1 "${OUT_DIR}"/best_coco_bbox_mAP_epoch_*.pth 2>/dev/null | tail -1 || true)
if [ -z "${BEST}" ]; then
  echo "[$(date '+%F %T')] ✗ 找不到 best ckpt，中止。"
  exit 1
fi
echo "[$(date '+%F %T')] best ckpt: ${BEST}"

# 緩衝，確保 GPU 記憶體釋放
sleep 30

echo "[$(date '+%F %T')] 啟動 4ch video inference..."
bash "${PROJECT_DIR}/run_infer_12th_v13_4ch_video.sh"
echo "[$(date '+%F %T')] 已送出 inference（背景執行，見 ${OUT_DIR}/video_inference.log）。"
