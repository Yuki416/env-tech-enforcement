#!/usr/bin/env bash
# YOLO26l 訓練（YOLO_baseline_定案規劃.md 步驟 3）
#
# 用法：
#   正式訓練：  bash train_yolo26l_v17.sh <seed>            # seed = 0、1、2
#   煙霧測試：  EPOCHS=1 FRACTION=0.1 NAME=smoke bash train_yolo26l_v17.sh 0
#
# 背景執行（log 寫到 HDD）：
#   nohup bash train_yolo26l_v17.sh 0 > /path/to/data/baseline_yolo26l_v17/logs/train_seed0.log 2>&1 &
#   tail -f /path/to/data/baseline_yolo26l_v17/logs/train_seed0.log
#
# 設定依定案規劃：imgsz 1024（對齊第 16 次）、100 epoch、其餘 Ultralytics 官方預設；
# patience=100 等於關掉提早停止；save_period=1 每個 epoch 都存權重，步驟 4 逐一評估挑最佳 epoch。
set -euo pipefail

SEED="${1:?用法：bash train_yolo26l_v17.sh <seed>}"
EPOCHS="${EPOCHS:-100}"
FRACTION="${FRACTION:-1.0}"
NAME="${NAME:-seed${SEED}}"
BATCH="${BATCH:-16}"

HDD=/path/to/data
OUT_HOST=$HDD/baseline_yolo26l_v17
mkdir -p "$OUT_HOST/weights_pretrained"

# 容器以一般使用者身分執行，避免在 HDD 留下 root 擁有的檔案。
# 工作目錄設在 weights_pretrained/，Ultralytics 下載的預訓練權重（yolo26l.pt、AMP 檢查用的 yolo26n.pt）會落在這裡。
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v $HDD/coco:/workspace/datasets/coco \
  -v $HDD:/workspace/hdd2_storage \
  -w /workspace/hdd2_storage/baseline_yolo26l_v17/weights_pretrained \
  yolo-baseline:latest \
  yolo detect train \
    model=yolo26l.pt \
    data=/workspace/datasets/coco/yolo_v17_own_dynamic_head10_manual/data.yaml \
    imgsz=1024 \
    epochs="$EPOCHS" \
    patience=100 \
    batch="$BATCH" \
    device=0,1 \
    workers=8 \
    save_period=1 \
    seed="$SEED" \
    deterministic=True \
    fraction="$FRACTION" \
    project=/workspace/hdd2_storage/baseline_yolo26l_v17 \
    name="$NAME" \
    exist_ok=False

echo "TRAIN_DONE seed=$SEED name=$NAME"
