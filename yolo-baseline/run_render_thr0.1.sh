#!/usr/bin/env bash
# ⚠️ 歷史紀錄（2026-10-02 起停用）：這是一對多頭＋NMS 版當時的執行方式，留著對照用。
#    裡面寫的是搬移前的路徑（結果已搬到 baseline_yolo26l_v17/nms/），也沒有設 YOLO_HEAD。
#    端到端重跑改用 rerun_*.sh，見 ../YOLO_baseline_端到端重跑規劃.md。
echo "此腳本已停用（歷史紀錄），請改用 rerun_*.sh" >&2; exit 1
# 門檻 0.1 的影片（3 個 seed）＋ 0.1／0.3／0.5 三組數字表。
# 0.1 接近 YOLO26l 在 valid 上 litter 的最佳 F1 門檻（0.11、0.12、0.11），看 YOLO 自己最有利的情況。
set -euo pipefail
cd "$(dirname "$0")"
L=/path/to/data/baseline_yolo26l_v17/logs
bash run_in_container.sh 0 metrics_at_thresholds.py --out /workspace/hdd2_storage/baseline_yolo26l_v17/thresholds_0.1_0.3_0.5 > $L/metrics_thr0.1_0.3_0.5.log 2>&1
grep -E "METRICS_THR_DONE|Error|assert" $L/metrics_thr0.1_0.3_0.5.log
bash run_in_container.sh 0 render_videos_from_pkl.py --seed 0 --thr 0.1 > $L/render_seed0_thr0.1.log 2>&1 &
bash run_in_container.sh 1 render_videos_from_pkl.py --seed 1 --thr 0.1 > $L/render_seed1_thr0.1.log 2>&1 &
wait
bash run_in_container.sh 0 render_videos_from_pkl.py --seed 2 --thr 0.1 > $L/render_seed2_thr0.1.log 2>&1
grep -hE "RENDER_DONE|Error|assert" $L/render_seed*_thr0.1.log
echo RENDER_THR01_DONE
