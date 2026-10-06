#!/bin/bash
# Co-DETR 第 11 次訓練 — v13 + Background Subtraction 4ch（單卡 GPU0）
# 單類別 litter-only，輸入 RGB+changemap 4 通道。conv1 已 inflate(平均初始化)。
# 對照 baseline = v12 best epoch 17（純 RGB）。

CONFIG="/workspace/project/configs/co_dino_chiayi_v13_4ch_litter_36e.py"
LOG="/path/to/data/11th_expt_v13_bgs_4ch_litter_36e/train.log"
mkdir -p "$(dirname "$LOG")"

echo "======================================"
echo " Co-DETR 第 11 次訓練 單卡（GPU0）v13 BGS 4ch"
echo " 輸入: RGB + changemap (4ch) | batch_size: 1"
echo " Log: $LOG"
echo "======================================"

nohup docker run --rm --gpus '"device=0"' \
  --ipc=host --shm-size=16g \
  -v /path/to/repo/co-detr:/workspace/project \
  -v /path/to/data/coco:/workspace/datasets/coco \
  -v /path/to/data:/workspace/hdd2_storage \
  -e PYTHONPATH=/opt/mmdetection:/workspace/project \
  -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 \
  co-detr:latest \
  python /opt/mmdetection/tools/train.py \
    "$CONFIG" \
    --cfg-options train_dataloader.batch_size=1 val_dataloader.batch_size=1 \
  > "$LOG" 2>&1 &

echo "背景執行中，PID: $!"
echo "查看進度： tail -f $LOG"
