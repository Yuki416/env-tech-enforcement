#!/usr/bin/env bash
# 端到端重跑步驟 0 與 0'（見 ../YOLO_baseline_端到端重跑規劃.md 3-3）。
#   0 ：YOLO_HEAD=nms 重算 seed 0（epoch 23）的 dynamic166、static594，輸出到 _repro_check_nms/（不寫進 nms/），
#       之後在主機上和 nms/seed0/test_eval/*_metrics.json 逐位元比對。
#   0'：verify_head.py 分別在 nms、e2e 下確認用的是哪個頭。
# 用法：nohup bash rerun_step0.sh > /path/to/data/baseline_yolo26l_v17/e2e/logs/step0.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
C=/workspace/hdd2_storage/baseline_yolo26l_v17
CK=$C/seed0/weights/epoch22.pt                      # seed 0 選中的 epoch 23（檔名從 0 起算）
echo "=== 步驟 0 開始 $(date '+%F %T') ==="
for SP in dynamic166 static594; do
  YOLO_HEAD=nms bash run_in_container.sh 0 predict_to_pkl.py --ckpt $CK --split $SP \
    --out $C/_repro_check_nms --tag "YOLO26l seed0 ep23 $SP" --device 0 2>&1 | grep -vE "Settings|settings|^\s*$"
done
echo "=== 步驟 0' 開始 $(date '+%F %T') ==="
for H in nms e2e; do
  YOLO_HEAD=$H bash run_in_container.sh 0 verify_head.py --ckpt $CK 2>&1 | grep -vE "Settings|settings|^\s*$"
done
echo "STEP0_DONE $(date '+%F %T')"
