#!/usr/bin/env bash
# 端到端重跑：影片（門檻 0.1 / 0.3 / 0.5 × 3 個 seed，共 9 套）與 metrics_thr0.1_0.3_0.5 數字表（見 ../YOLO_baseline_端到端重跑規劃.md 3-3）。
# 影片要用步驟 5 的 full_preds 與步驟 6 的 threshold_eval.json（摘要裡寫 best-F1 門檻），所以在步驟 6 之後跑。
# 數字表的 clip 分類一節讀影片產生的 video_classification.csv，所以最後才產生。
# 兩張 GPU 各跑一條佇列（畫影片主要吃 CPU 與硬碟，GPU 只是容器需要）。
# 用法：YOLO_HEAD=e2e nohup bash rerun_videos.sh > /path/to/data/baseline_yolo26l_v17/e2e/logs/videos.log 2>&1 &
set -euo pipefail
: "${YOLO_HEAD:?必須設定 YOLO_HEAD=e2e 或 nms}"
cd "$(dirname "$0")"
HOST=/path/to/data/baseline_yolo26l_v17
L=$HOST/$YOLO_HEAD/logs
R=/workspace/hdd2_storage/baseline_yolo26l_v17/$YOLO_HEAD
q() {  # q <GPU> <seed:thr>...
  local G=$1; shift
  for st in "$@"; do
    bash run_in_container.sh "$G" render_videos_from_pkl.py --seed "${st%%:*}" --thr "${st##*:}" > "$L/render_seed${st%%:*}_thr${st##*:}.log" 2>&1
    grep -h "RENDER_DONE" "$L/render_seed${st%%:*}_thr${st##*:}.log"
  done
}
echo "=== 影片開始 $(date '+%F %T') ==="
q 0 0:0.1 0:0.3 0:0.5 1:0.1 1:0.3 &
P0=$!
q 1 1:0.5 2:0.1 2:0.3 2:0.5 &
P1=$!
wait $P0; wait $P1
echo "=== 數字表 $(date '+%F %T') ==="
bash run_in_container.sh 0 metrics_at_thresholds.py --out $R/thresholds_0.1_0.3_0.5 > "$L/metrics_thr0.1_0.3_0.5.log" 2>&1
grep -h "METRICS_THR_DONE" "$L/metrics_thr0.1_0.3_0.5.log"
echo "VIDEOS_DONE $(date '+%F %T')"
