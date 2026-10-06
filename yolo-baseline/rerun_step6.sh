#!/usr/bin/env bash
# 端到端重跑步驟 6：best-F1 門檻（valid 決定、套到 test）與 PR 曲線（見 ../YOLO_baseline_端到端重跑規劃.md 3-3）。
# 第 16 次 Co-DETR 的 threshold_eval.json 兩版共用，直接沿用 codetr16_valid/。
# 結果：<YOLO_HEAD>/seed<N>/test_eval/threshold_eval.json、<YOLO_HEAD>/report_figs/pr_{litter,garbage}_{dynamic166,static594}.png
# 用法：YOLO_HEAD=e2e nohup bash rerun_step6.sh > /path/to/data/baseline_yolo26l_v17/e2e/logs/step6.log 2>&1 &
set -euo pipefail
: "${YOLO_HEAD:?必須設定 YOLO_HEAD=e2e 或 nms}"
cd "$(dirname "$0")"
C=/workspace/hdd2_storage/baseline_yolo26l_v17
R=$C/$YOLO_HEAD
for S in 0 1 2; do
  T=$R/seed$S/test_eval
  bash run_in_container.sh 0 eval_with_threshold.py --name "YOLO26l $YOLO_HEAD seed$S" --valid-pkl $T/valid.pkl \
    --test-pkl dynamic166=$T/dynamic166.pkl --test-pkl static594=$T/static594.pkl --out $T/threshold_eval.json 2>&1 | grep -vE "Settings|settings|^\s*$"
done
for c in litter garbage; do
  bash run_in_container.sh 0 plot_pr_curve.py --cls $c --out $R/report_figs "Co-DETR 16th=$C/codetr16_valid/threshold_eval.json" \
    "YOLO26l $YOLO_HEAD seed0=$R/seed0/test_eval/threshold_eval.json" "YOLO26l $YOLO_HEAD seed1=$R/seed1/test_eval/threshold_eval.json" \
    "YOLO26l $YOLO_HEAD seed2=$R/seed2/test_eval/threshold_eval.json" 2>&1 | grep saved
done
echo "STEP6_DONE $(date '+%F %T')"
