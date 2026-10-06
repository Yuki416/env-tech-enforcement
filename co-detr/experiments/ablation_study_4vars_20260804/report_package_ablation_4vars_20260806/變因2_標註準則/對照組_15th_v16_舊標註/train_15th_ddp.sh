#!/bin/bash
# Co-DETR v16 動態版 雙卡 DDP 訓練腳本（backup server ccu2）
# 第十五次實驗：v16 = 121 部影片的『動態子集』（配方與第十三次 v14 逐字相同，只有資料變多）
#   train 3009 幀 / 99 clips（含 ~46% negative 填充幀）；valid 488 幀 / 14 clips（與第十三次完全相同）
#   test 用 v16 完整版 655 幀 / 8 clips（不過濾）。
#   ★ 對照 = 第十三次 v14（train 1237 幀 / 44 clips）。第十三次訓練集是本次的嚴格子集。
#     單一變因 = 訓練資料規模（2.43×）。
# GPU 0 + GPU 1，每卡 batch_size=1，等效 batch_size=2（與第十三次相同）

CONFIG="/workspace/project/configs/co_dino_chiayi_v16_own_dynamic_36e.py"
NUM_GPUS=2
# 結果直接寫 HDD（host 路徑，nohup 在 host 執行）；results/ 有 symlink 指向此處供瀏覽
LOG="/path/to/data/15th_expt_v16_own_dynamic_121vids_36e/train.log"

mkdir -p "$(dirname "$LOG")"

echo "======================================"
echo " Co-DETR v16 動態版 DDP 訓練（第十五次）"
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
