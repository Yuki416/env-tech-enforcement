#!/usr/bin/env bash
# 端到端重跑步驟 7（bootstrap、逐 clip 表）與步驟 8（速度）（見 ../YOLO_baseline_端到端重跑規劃.md 3-3）。
# 速度條件照舊：GPU 1（RTX 4090）、FP32、batch 1、dynamic166 依 id 排序第 11～110 張、前 10 張暖機；權重用 seed 0 選中的 epoch。
# 先跑速度（要獨佔 GPU 1、量測期間不能有別的工作），再跑 bootstrap 與逐 clip 表。
# 用法：YOLO_HEAD=e2e nohup bash rerun_step7_8.sh > /path/to/data/baseline_yolo26l_v17/e2e/logs/step7_8.log 2>&1 &
set -euo pipefail
: "${YOLO_HEAD:?必須設定 YOLO_HEAD=e2e 或 nms}"
cd "$(dirname "$0")"
HOST=/path/to/data/baseline_yolo26l_v17
C=/workspace/hdd2_storage/baseline_yolo26l_v17
R=$C/$YOLO_HEAD
B0=$(python3 -c "import json;print(json.load(open('$HOST/$YOLO_HEAD/seed0/epoch_select/best_epoch.json'))['best_epoch'])")
echo "=== 步驟 8 速度（GPU 1、seed 0 epoch $B0）$(date '+%T') ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
mkdir -p $HOST/$YOLO_HEAD/speed
bash run_in_container.sh 1 benchmark_speed.py --ckpt $C/seed0/weights/epoch$((B0-1)).pt --out $R/speed/yolo26l.json 2>&1 | grep -vE "Settings|settings|^\s*$"
echo "=== 步驟 7 bootstrap、逐 clip 表 $(date '+%T') ==="
bash run_in_container.sh 1 bootstrap_ci.py --out $R/bootstrap --n 1000 > $HOST/$YOLO_HEAD/logs/bootstrap.log 2>&1 &
PB=$!
bash run_in_container.sh 0 per_clip_table.py --out $R/per_clip 2>&1 | grep -vE "Settings|settings|^\s*$" | tail -3
wait $PB
tail -5 $HOST/$YOLO_HEAD/logs/bootstrap.log
echo "STEP7_8_loc7 $(date '+%F %T')"
