#!/usr/bin/env bash
# 縮放實驗：依序跑 YOLO（三個 seed × 4 種尺寸）與第 16 次 Co-DETR（4 種尺寸），都在同一張 GPU 上。
# 用法：nohup bash run_scale_sweep.sh 0 > /path/to/data/baseline_yolo26l_v17/logs/scale_sweep.log 2>&1 &
set -euo pipefail
GPU="${1:?用法：bash run_scale_sweep.sh <GPU 編號>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
mkdir -p /path/to/data/baseline_yolo26l_v17/e2e/scale_sweep
YOLO_HEAD=e2e bash run_in_container.sh "$GPU" scale_sweep_yolo.py
bash scale_sweep_codetr16.sh "$GPU"
echo RUN_SCALE_SWEEP_DONE
