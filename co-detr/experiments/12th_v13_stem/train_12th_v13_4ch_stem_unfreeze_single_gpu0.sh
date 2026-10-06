#!/bin/bash
# Co-DETR 第 12 次訓練 — v13 + BGS 4ch，解凍 stem 讓遮罩通道學得到（單卡 GPU0）
# 資料集與第 11 次完全相同（v13 dynamic test）。改動：frozen_stages=-1 / conv1 lr_mult=1.0
# / norm_eval=False / Conv1Ch4NormLogger 探針。對照 baseline 仍是 v12 best epoch 17（純 RGB）。
# 解凍 stem+layer1 後反傳記憶體更吃緊 → 沿用 bs=1。

CONFIG="/workspace/project/configs/co_dino_chiayi_v13_4ch_litter_12th_36e.py"
LOG="/path/to/data/12th_expt_v13_bgs_4ch_stem_unfreeze_36e/train.log"
mkdir -p "$(dirname "$LOG")"

echo "======================================"
echo " Co-DETR 第 12 次訓練 單卡（GPU0）v13 BGS 4ch（解凍 stem）"
echo " 輸入: RGB + changemap (4ch) | batch_size: 1"
echo " Log: $LOG"
echo "======================================"

nohup docker run --rm --name co-detr-train-12th --gpus '"device=0"' \
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
