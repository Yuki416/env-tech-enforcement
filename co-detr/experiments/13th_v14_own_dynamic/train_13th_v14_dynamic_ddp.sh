#!/bin/bash
# Co-DETR v14 動態版 雙卡 DDP 訓練腳本（backup server ccu2）
# 第十三次實驗：v14 = 我們自己 v11 資料的『動態子集』（litter+garbage 動態種子 + 連續片段擷取）
#   train 2192 幀 / valid 770 幀（含 ~27-32% negative 填充幀）；test 沿用原始 v11 靜止版。
# GPU 0 + GPU 1，每卡 batch_size=1，等效 batch_size=2（與第九次 v11 相同）

CONFIG="/workspace/project/configs/co_dino_chiayi_v14_own_dynamic_36e.py"
NUM_GPUS=2
# 結果直接寫 HDD（host 路徑，nohup 在 host 執行）；results/ 有 symlink 指向此處供瀏覽
LOG="/path/to/data/13th_expt_v14_own_dynamic_36e/train.log"

mkdir -p "$(dirname "$LOG")"

echo "======================================"
echo " Co-DETR v14 動態版 DDP 訓練（第十三次）"
echo " GPUs: 2x RTX 4090（等效 bs=2）"
echo " Config: $CONFIG"
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
