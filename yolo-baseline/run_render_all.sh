#!/usr/bin/env bash
# ⚠️ 歷史紀錄（2026-10-02 起停用）：這是一對多頭＋NMS 版當時的執行方式，留著對照用。
#    裡面寫的是搬移前的路徑（結果已搬到 baseline_yolo26l_v17/nms/），也沒有設 YOLO_HEAD。
#    端到端重跑改用 rerun_*.sh，見 ../YOLO_baseline_端到端重跑規劃.md。
echo "此腳本已停用（歷史紀錄），請改用 rerun_*.sh" >&2; exit 1
# 3 個 seed × 2 種信心分數門檻（0.3、0.5）的影片，兩套並行。
set -euo pipefail
cd "$(dirname "$0")"
L=/path/to/data/baseline_yolo26l_v17/logs
for S in 0 1 2; do
  bash run_in_container.sh 0 render_videos_from_pkl.py --seed $S --thr 0.3 > $L/render_seed${S}_thr0.3.log 2>&1 &
  bash run_in_container.sh 1 render_videos_from_pkl.py --seed $S --thr 0.5 > $L/render_seed${S}_thr0.5.log 2>&1 &
  wait
  grep -hE "RENDER_DONE|Error|assert" $L/render_seed${S}_thr0.3.log $L/render_seed${S}_thr0.5.log
done
echo RENDER_ALL_DONE
