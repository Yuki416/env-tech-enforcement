#!/bin/bash
# Co-DETR v17 動態版（HEAD=10 + 手動微調）雙卡 DDP 訓練腳本（backup server ccu2）
# 第十六次實驗：資料集 = chiayi_v17_own_dynamic_litter_head10_manual
#   train 2494 幀 / 99 clips（含 negative 填充幀）；valid 373 幀 / 15 clips；
#   test 用 v17 完整版 655 幀 / 8 clips（不過濾）。
#   配方沿用第十五次，**唯一實質變動 = 資料集**（v16→v17 + HEAD30→10 + 8 部手動微調）。
#   ★ 主判定 = test_static594 對比第十三次 litter AP@0.5 = 0.254（見實驗夾 README）。
# GPU 0 + GPU 1，每卡 batch_size=1，等效 batch_size=2（與第十三/十五次相同）。

CONFIG="/workspace/project/configs/co_dino_chiayi_v17_own_dynamic_head10_manual_36e.py"
NUM_GPUS=2
# 結果直接寫 HDD（host 路徑，nohup 在 host 執行）；results/ 有 symlink 指向此處供瀏覽
LOG="/path/to/data/16th_expt_v17_own_dynamic_122vids_36e/train.log"

mkdir -p "$(dirname "$LOG")"

echo "======================================"
echo " Co-DETR v17 動態版 DDP 訓練（第十六次）"
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
