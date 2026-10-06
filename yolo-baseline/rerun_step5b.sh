#!/usr/bin/env bash
# 端到端重跑步驟 5b：同一個 checkpoint 只換頭的對照（見 ../YOLO_baseline_端到端重跑規劃.md 3-3）。
# 端到端版選中的 epoch 若和 NMS 版不同，就用 YOLO_HEAD=nms 在「端到端版選中的 epoch」上跑 dynamic166、static594，
# 這樣同一個 checkpoint 兩種頭都有 test 結果；epoch 相同就直接沿用 nms/seed<N>/test_eval，不必跑。
# 結果：e2e/head_compare/seed<N>/nms_ep<B>/{dynamic166,static594}{.pkl,_metrics.json}
# 用法：nohup bash rerun_step5b.sh > /path/to/data/baseline_yolo26l_v17/e2e/logs/step5b.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")"
HOST=/path/to/data/baseline_yolo26l_v17
C=/workspace/hdd2_storage/baseline_yolo26l_v17
be() { python3 -c "import json;print(json.load(open('$HOST/$1/seed$2/epoch_select/best_epoch.json'))['best_epoch'])"; }
for S in 0 1 2; do
  BE=$(be e2e $S); BN=$(be nms $S)
  if [ "$BE" = "$BN" ]; then echo "seed $S：兩版都選 epoch $BE，沿用 nms/seed$S/test_eval"; continue; fi
  echo "=== seed $S：e2e 選 epoch $BE、nms 選 epoch $BN → 在 epoch $BE 上補跑 nms 頭 $(date '+%T') ==="
  for SP in dynamic166 static594; do
    YOLO_HEAD=nms bash run_in_container.sh 0 predict_to_pkl.py --ckpt $C/seed$S/weights/epoch$((BE-1)).pt --split $SP \
      --out $C/e2e/head_compare/seed$S/nms_ep$BE --tag "YOLO26l nms seed$S ep$BE $SP" --device 0 2>&1 | grep -E "=====|^litter|^garbage|^car|→"
  done
done
echo "STEP5B_DONE $(date '+%F %T')"
