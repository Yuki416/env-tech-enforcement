#!/bin/bash
# Co-DETR v15 BGS 4ch 雙卡 DDP 訓練腳本（backup server ccu2）
# 第 14 次實驗：v15 = 第 13 次 v14 動態子集「同影像同標註」+ pre-roll BGS changemap 第 4 通道。
#   對照 baseline = 第 13 次 v14（純 RGB）。4ch 機制沿用第 12 次（解凍 stem + conv1 大 lr）。
# GPU 0 + GPU 1，每卡 batch_size=1，等效 batch_size=2（與第 12/13 次相同；dryrun 實測 bs=1 約 15.4GB）。

CONFIG="/workspace/project/configs/co_dino_chiayi_v15_own_dynamic_bgs_4ch_36e.py"
NUM_GPUS=2
LOG="/path/to/data/14th_expt_v15_own_dynamic_bgs_4ch_36e/train.log"

mkdir -p "$(dirname "$LOG")"

echo "======================================"
echo " Co-DETR v15 BGS 4ch DDP 訓練（第 14 次）"
echo " GPUs: 2x RTX 4090（各 bs=1，等效 bs=2）"
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
  -e PYTHONUTF8=1 \
  `# PYTHONUTF8=1：強制 open() 用 UTF-8，免得 CUDA/torch 重設 locale 後 pycocotools 讀含中文檔名的 json 爆 ascii` \
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
