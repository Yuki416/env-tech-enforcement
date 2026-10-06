#!/bin/bash
# Co-DETR 第 17 次訓練 雙卡 DDP（backup server ccu2）
# 資料集：chiayi_v17_own_dynamic_litter_head10_manual（= 第16次同一份 V17，不另產新檔）
#   只取 litter + garbage 兩類（config metainfo 子集），其餘 car/person/motorcycle 標註自動丟棄。
# 方法：完全照第 10 次配方（plain FocalLoss/CE、backbone lr_mult=0.1、milestones[24,33]、無 ClassBalanced）；
#       兩類同權（無 class weighting）；save_best=coco/litter_precision。
# GPU 0 + GPU 1，每卡 batch_size=1，等效 batch_size=2。
# work_dir 寫到 HDD 實體路徑（/workspace/hdd2_storage/17th_expt_...）。

CONFIG="/workspace/project/configs/co_dino_chiayi_v17_own_dynamic_2cls_litter_garbage_36e.py"
NUM_GPUS=2
LOG="/path/to/data/17th_expt_v17_own_dynamic_2cls_36e/train.log"

mkdir -p "$(dirname "$LOG")"

echo "======================================"
echo " Co-DETR 第 17 次訓練 DDP（v17 動態版，2 類 litter+garbage）"
echo " GPUs: 2x RTX 4090（等效 bs=2）"
echo " Config: $CONFIG"
echo " Log: $LOG"
echo "======================================"

# -e PYTHONUTF8=1：資料集含中文檔名，pycocotools 用 locale 預設編碼 open() json，
#   CUDA/torch 可能把 locale 重設成 ascii → UnicodeDecodeError。強制 UTF-8 是零成本保險。
nohup docker run --rm --gpus all \
  --ipc=host --shm-size=16g \
  -v /path/to/repo/co-detr:/workspace/project \
  -v /path/to/data/coco:/workspace/datasets/coco \
  -v /path/to/data:/workspace/hdd2_storage \
  -e PYTHONPATH=/opt/mmdetection:/workspace/project \
  -e LANG=C.UTF-8 \
  -e LC_ALL=C.UTF-8 \
  -e PYTHONUTF8=1 \
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
