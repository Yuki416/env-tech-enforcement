#!/bin/bash
# Co-DETR 第 10 次訓練 — 單卡（GPU0）訓練腳本（backup server ccu2）
# 改用單卡原因：共享機器雙卡長時間等不到（>29h），GPU0 有 ~20GB 空閒，足夠 bs=1。
# 資料集：v12 (chiayi_v12_other_team_litter_annotation)，單類別 litter-only
# 只開放 GPU0 給容器（--gpus device=0），用 tools/train.py 單卡訓練（免 DDP）

CONFIG="/workspace/project/configs/co_dino_chiayi_v12_litter_only_36e.py"
LOG="/path/to/data/10th_expt_v12_other_team_litter_36e/train.log"

mkdir -p "$(dirname "$LOG")"

echo "======================================"
echo " Co-DETR 第 10 次訓練 單卡（GPU0）（v12 litter-only）"
echo " GPU: 僅 GPU0"
echo " batch_size: 1"
echo " Log: $LOG"
echo "======================================"

nohup docker run --rm --gpus '"device=0"' \
  --ipc=host --shm-size=16g \
  -v /path/to/repo/co-detr:/workspace/project \
  -v /path/to/data/coco:/workspace/datasets/coco \
  -v /path/to/data:/workspace/hdd2_storage \
  -e PYTHONPATH=/opt/mmdetection:/workspace/project \
  -e LANG=C.UTF-8 \
  -e LC_ALL=C.UTF-8 \
  co-detr:latest \
  python /opt/mmdetection/tools/train.py \
    "$CONFIG" \
    --cfg-options train_dataloader.batch_size=1 val_dataloader.batch_size=1 \
  > "$LOG" 2>&1 &

echo "背景執行中，PID: $!"
echo ""
echo "查看進度："
echo "  tail -f $LOG"
