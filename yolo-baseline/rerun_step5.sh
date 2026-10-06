#!/usr/bin/env bash
# 端到端重跑步驟 5：用步驟 4 選中的 epoch 推論（見 ../YOLO_baseline_端到端重跑規劃.md 3-3）。
#   test_eval/ ：dynamic166、static594、valid（步驟 6 要用 valid.pkl）
#   full_preds/：train、valid、test_full（影片用）
#   step5_checks/：① check_eval_consistency（Ultralytics 內建評估，同一個頭）② seed 0 的 dynamic166 框畫回原圖
#   最後 score_floor.py 量實際存檔下限（驗收 ③）
# seed 0、1 在兩張 GPU 平行，seed 2 接在 seed 0 後面。
# 用法：YOLO_HEAD=e2e nohup bash rerun_step5.sh > /path/to/data/baseline_yolo26l_v17/e2e/logs/step5.log 2>&1 &
set -euo pipefail
: "${YOLO_HEAD:?必須設定 YOLO_HEAD=e2e 或 nms}"
cd "$(dirname "$0")"
HOST=/path/to/data/baseline_yolo26l_v17
C=/workspace/hdd2_storage/baseline_yolo26l_v17
R=$C/$YOLO_HEAD
L=$HOST/$YOLO_HEAD/logs
V18=/workspace/datasets/coco/chiayi_v18_own_dynamic_bgs_122vids

seed_job() {  # seed_job <GPU> <seed>
  local G=$1 S=$2
  local B; B=$(python3 -c "import json;print(json.load(open('$HOST/$YOLO_HEAD/seed$S/epoch_select/best_epoch.json'))['best_epoch'])")
  local CK=$C/seed$S/weights/epoch$((B-1)).pt
  echo "=== seed $S（epoch $B，GPU $G）開始 $(date '+%F %T') ==="
  for SP in dynamic166 static594 valid; do
    bash run_in_container.sh "$G" predict_to_pkl.py --ckpt "$CK" --split $SP --out $R/seed$S/test_eval \
      --tag "YOLO26l $YOLO_HEAD seed$S ep$B $SP" --device 0 > "$L/step5_seed${S}_test_eval_$SP.log" 2>&1
  done
  for SP in train valid test_full; do
    bash run_in_container.sh "$G" predict_to_pkl.py --ckpt "$CK" --split $SP --out $R/seed$S/full_preds \
      --tag "YOLO26l $YOLO_HEAD seed$S ep$B $SP" --device 0 > "$L/step5_seed${S}_full_preds_$SP.log" 2>&1
  done
  bash run_in_container.sh "$G" check_eval_consistency.py --ckpt "$CK" --out $R/seed$S/step5_checks --device 0 \
    > "$L/step5_seed${S}_consistency.log" 2>&1
  if [ "$S" = 0 ]; then
    bash run_in_container.sh "$G" draw_pkl_boxes.py --pkl $R/seed0/test_eval/dynamic166.pkl \
      --ann $V18/annotations/test_dynamic166.json --img-dir $V18/test --out $R/seed0/step5_checks/boxes_dynamic166 \
      > "$L/step5_seed0_draw.log" 2>&1
  fi
  echo "=== seed $S 結束 $(date '+%F %T') ==="
}
( seed_job 0 0; seed_job 0 2 ) &
P0=$!
( seed_job 1 1 ) &
P1=$!
wait $P0; wait $P1
bash run_in_container.sh 0 score_floor.py > "$L/step5_score_floor.log" 2>&1
grep -h "=====\|^litter\|^garbage" "$L"/step5_seed*_test_eval_*.log
grep -h -A6 "UL AP50" "$L"/step5_seed*_consistency.log
cat "$HOST/$YOLO_HEAD/score_floor.md"
echo "STEP5_DONE $(date '+%F %T')"
