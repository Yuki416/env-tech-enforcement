#!/usr/bin/env bash
# 在 yolo-baseline 容器裡執行本資料夾的 Python 腳本（步驟 4 以後共用）。
#
# 用法：bash run_in_container.sh <GPU 編號> <腳本與參數...>
#   例：YOLO_HEAD=e2e bash run_in_container.sh 0 select_best_epoch.py --seed 0
#   YOLO_HEAD（e2e 或 nms）必填，會傳進容器，決定用哪個偵測頭與結果寫到哪一版的資料夾（見 yolo_eval_common.py）。
#
# 掛載：
#   /workspace/datasets/coco   ← /path/to/data/coco（資料集）
#   /workspace/hdd2_storage    ← /path/to/data（訓練結果）
#   /workspace/yolo            ← 本資料夾（腳本）
#   /workspace/codetr_exp      ← co-detr/experiments（唯讀；取用 metrics_from_preds.py，不修改）
set -euo pipefail
GPU="${1:?用法：YOLO_HEAD=<e2e|nms> bash run_in_container.sh <GPU> <script> [args...]}"; shift
: "${YOLO_HEAD:?必須設定 YOLO_HEAD=e2e 或 nms}"
HERE="$(cd "$(dirname "$0")" && pwd)"
HDD=/path/to/data
docker run --rm --gpus "\"device=${GPU}\"" --ipc=host --shm-size=8g \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -e YOLO_HEAD="$YOLO_HEAD" \
  -v $HDD/coco:/workspace/datasets/coco \
  -v $HDD:/workspace/hdd2_storage \
  -v "$HERE":/workspace/yolo \
  -v "$HERE/../../co-detr/experiments":/workspace/codetr_exp:ro \
  -w /workspace/yolo \
  yolo-baseline:latest python "$@"
