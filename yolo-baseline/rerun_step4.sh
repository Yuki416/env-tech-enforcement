#!/usr/bin/env bash
# 端到端重跑步驟 4：3 個 seed × 100 個 checkpoint 在 valid 上重新挑 epoch（規則不變：valid litter AP@[.5:.95] 最高）。
# 兩張 GPU：seed 0 用 GPU 0、seed 1 用 GPU 1 平行，seed 2 接在 seed 0 後面跑在 GPU 0。
# 結果：<ROOT>/<YOLO_HEAD>/seed<N>/epoch_select/；各 seed 的 log 在 <ROOT>/<YOLO_HEAD>/logs/select_best_epoch_seed<N>.log
# 用法：YOLO_HEAD=e2e nohup bash rerun_step4.sh > /path/to/data/baseline_yolo26l_v17/e2e/logs/step4.log 2>&1 &
set -euo pipefail
: "${YOLO_HEAD:?必須設定 YOLO_HEAD=e2e 或 nms}"
cd "$(dirname "$0")"
L=/path/to/data/baseline_yolo26l_v17/$YOLO_HEAD/logs
mkdir -p "$L"
run() {  # run <GPU> <seed>
  echo "=== seed $2 開始（GPU $1）$(date '+%F %T') ==="
  bash run_in_container.sh "$1" select_best_epoch.py --seed "$2" --device 0 > "$L/select_best_epoch_seed$2.log" 2>&1
  echo "=== seed $2 結束 $(date '+%F %T') ==="
  grep -A12 '"rule"' "$L/select_best_epoch_seed$2.log" || true
}
( run 0 0; run 0 2 ) &
P0=$!
( run 1 1 ) &
P1=$!
wait $P0; wait $P1
grep -c "SELECT_BEST_EPOCH_DONE" "$L"/select_best_epoch_seed{0,1,2}.log
echo "STEP4_DONE $(date '+%F %T')"
