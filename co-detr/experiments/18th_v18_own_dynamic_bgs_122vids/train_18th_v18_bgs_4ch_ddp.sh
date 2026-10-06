#!/bin/bash
# Co-DETR 第 18 次實驗：v18 = v17 手動動態子集 + Background Subtraction 4ch，雙卡 DDP 訓練。
# 資料集 = chiayi_v18_own_dynamic_bgs_122vids（= 第16次 v17 手動版同影像同標註 + 平行 changemap）。
#   train 2494 幀/99 clips、valid 373/15、test 655/8；changemap 3522 張與 jpg 1:1。
# 單變因對照 = 第16次純 RGB。成敗以動態子集 test_dynamic166 為主判定、static594 副（訓練後另跑）。
#   4ch：backbone in_channels=4 + frozen_stages=-1 + norm_eval=True；conv1 lr 1.0x、backbone 其餘 0.01x(=同16th)。
#   dryrun（dryrun_18th_v18.py）已 DRYRUN_ALL_PASS。
# GPU 0 + GPU 1，每卡 batch_size=1，等效 batch_size=2（與第13/15/16次相同；dryrun 峰值 15.4GB/卡）。

CONFIG="/workspace/project/configs/co_dino_chiayi_v18_own_dynamic_bgs_4ch_36e.py"
NUM_GPUS=2
# 結果直接寫 HDD（host 路徑，nohup 在 host 執行）；results/ 有 symlink 指向此處供瀏覽。
LOG="/path/to/data/18th_expt_v18_own_dynamic_bgs_4ch_36e/train.log"

mkdir -p "$(dirname "$LOG")"

echo "======================================"
echo " Co-DETR v18 BGS 4ch DDP 訓練（第十八次）"
echo " GPUs: 2x RTX 4090（每卡 bs=1，等效 bs=2）"
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
