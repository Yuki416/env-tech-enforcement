#!/usr/bin/env bash
# ⚠️ 歷史紀錄（2026-10-02 起停用）：這是一對多頭＋NMS 版當時的執行方式，留著對照用。
#    裡面寫的是搬移前的路徑（結果已搬到 baseline_yolo26l_v17/nms/），也沒有設 YOLO_HEAD。
#    端到端重跑改用 rerun_*.sh，見 ../YOLO_baseline_端到端重跑規劃.md。
echo "此腳本已停用（歷史紀錄），請改用 rerun_*.sh" >&2; exit 1
# seed 0 的步驟 4、5 驗收都通過後，依序完成 seed 1、2 的「訓練 → 挑 epoch → test 評估」。
# 用法：nohup bash run_seeds_1_2.sh > /path/to/data/baseline_yolo26l_v17/logs/run_seeds_1_2.log 2>&1 &
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG=/path/to/data/baseline_yolo26l_v17/logs
cd "$HERE"
for S in 1 2; do
  echo "=== seed $S：訓練 $(date '+%F %T') ==="
  bash train_yolo26l_v17.sh "$S" > "$LOG/train_seed${S}.log" 2>&1
  echo "=== seed $S：挑 epoch $(date '+%F %T') ==="
  bash run_in_container.sh 0 select_best_epoch.py --run /workspace/hdd2_storage/baseline_yolo26l_v17/seed$S --device 0 \
    > "$LOG/select_best_epoch_seed${S}.log" 2>&1
  BEST=$(python3 -c "import json;print(json.load(open('/path/to/data/baseline_yolo26l_v17/seed$S/epoch_select/best_epoch.json'))['best_epoch'])")
  echo "seed $S 選中 epoch $BEST"
  for SP in dynamic166 static594; do
    bash run_in_container.sh 0 predict_to_pkl.py \
      --ckpt /workspace/hdd2_storage/baseline_yolo26l_v17/seed$S/weights/epoch$((BEST-1)).pt \
      --split $SP --out /workspace/hdd2_storage/baseline_yolo26l_v17/seed$S/test_eval \
      --tag "YOLO26l seed$S ep$BEST $SP" --device 0 2>&1 | grep -vE "Settings|settings|^\s*$"
  done
done
echo "SEEDS_1_2_DONE $(date '+%F %T')"
