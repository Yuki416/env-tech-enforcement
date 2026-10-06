#!/usr/bin/env bash
# ⚠️ 歷史紀錄（2026-10-02 起停用）：這是一對多頭＋NMS 版當時的執行方式，留著對照用。
#    裡面寫的是搬移前的路徑（結果已搬到 baseline_yolo26l_v17/nms/），也沒有設 YOLO_HEAD。
#    端到端重跑改用 rerun_*.sh，見 ../YOLO_baseline_端到端重跑規劃.md。
echo "此腳本已停用（歷史紀錄），請改用 rerun_*.sh" >&2; exit 1
# 影片推論前置：三個 seed 在 train / valid / test_full 全部 frames 上推論，存成 pkl（設定與步驟 5 相同）。
# valid.pkl 步驟 6 已經產生過，這裡重新產生一次確保三個 split 用同一支程式、同一時間的結果。
set -euo pipefail
cd "$(dirname "$0")"
for S in 0 1 2; do
  BEST=$(python3 -c "import json;print(json.load(open('/path/to/data/baseline_yolo26l_v17/seed$S/epoch_select/best_epoch.json'))['best_epoch'])")
  for SP in train valid test_full; do
    echo "=== seed $S（epoch $BEST）$SP $(date '+%T') ==="
    bash run_in_container.sh $((S % 2)) predict_to_pkl.py \
      --ckpt /workspace/hdd2_storage/baseline_yolo26l_v17/seed$S/weights/epoch$((BEST-1)).pt \
      --split $SP --out /workspace/hdd2_storage/baseline_yolo26l_v17/seed$S/full_preds \
      --tag "YOLO26l seed$S ep$BEST $SP" --device 0 2>&1 | grep -E "=====|litter|garbage|→"
  done
done
echo PREDICT_ALL_DONE
