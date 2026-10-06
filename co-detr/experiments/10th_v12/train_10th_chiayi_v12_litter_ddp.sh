#!/bin/bash
# Co-DETR 第 10 次訓練 雙卡 DDP 訓練腳本（backup server ccu2）
# 資料集：v12 (chiayi_v12_other_team_litter_annotation)，單類別 litter-only
#   - train/valid: other team litter 標註（train 排除 2 部與 test 重疊影片，保留背景圖）
#   - test: chiayi_v11 test（litter+garbage 併成 litter）
# GPU 0 + GPU 1，每卡 batch_size=1，等效 batch_size=2
# work_dir 寫到 /workspace/hdd2_storage/10th_expt_v12_other_team_litter_36e（= HDD 實體資料夾）

CONFIG="/workspace/project/configs/co_dino_chiayi_v12_litter_only_36e.py"
NUM_GPUS=2
LOG="/path/to/data/10th_expt_v12_other_team_litter_36e/train.log"

mkdir -p "$(dirname "$LOG")"

echo "======================================"
echo " Co-DETR 第 10 次訓練 DDP（v12 litter-only）"
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
