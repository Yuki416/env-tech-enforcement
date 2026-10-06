#!/bin/bash
# Co-DETR v10 雙卡 DDP 訓練腳本（backup server）
# GPU 0 + GPU 1，每卡 batch_size=1，等效 batch_size=2

CONFIG="/workspace/project/configs/co_dino_chiayi_v10_36e_65vids.py"
NUM_GPUS=2
LOG="/path/to/repo/co-detr/results/chiayi_v10_36e_65vids/train.log"

echo "======================================"
echo " Co-DETR v10 DDP 訓練"
echo " GPUs: 2x RTX 4090"
echo " batch_size: 1 per GPU（等效 bs=2）"
echo " Log: $LOG"
echo "======================================"

nohup docker run --rm --gpus all \
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
    "$NUM_GPUS" \
    --cfg-options train_dataloader.batch_size=1 val_dataloader.batch_size=1 \
  > "$LOG" 2>&1 &

echo "背景執行中，PID: $!"
echo ""
echo "查看進度："
echo "  tail -f $LOG"
