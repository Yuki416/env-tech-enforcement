#!/bin/bash
# 消融實驗 變因① 處理組（靜止尾 statictail）雙卡 DDP 訓練（backup server ccu2）
#   資料集 = chiayi_v17_own_dynamic_head10_manual_statictail_ablation
#   train 3551 幀 / 99 clip（2494 對照 + 1057 靜止尾）；valid 373 幀（= 16th）；test 655 幀。
#   對照組 = 第 16 次（已訓練，best=ep23）。自變因 = 訓練集是否納入靜止尾。
#   ★ config 相對 16th 只改：data_root、max_keep_ckpts=-1（全留 36 顆）、work_dir/log。
# GPU 0 + GPU 1，每卡 batch_size=1，等效 batch_size=2（與 13/15/16 次相同）。

CONFIG="/workspace/project/configs/co_dino_chiayi_v17_statictail_ablation_36e.py"
NUM_GPUS=2
# 結果直接寫 HDD（host 路徑，nohup 在 host 執行）
LOG="/path/to/data/ablation_var1_statictail_36e/train.log"

mkdir -p "$(dirname "$LOG")"

echo "======================================"
echo " 消融 變因① 靜止尾 statictail DDP 訓練"
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
