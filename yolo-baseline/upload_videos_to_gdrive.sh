#!/usr/bin/env bash
# 把 YOLO26l 端到端版（e2e）的影片推論結果上傳到 Google Drive：gdrive:Co-DETR_training/baseline_yolo26l_v17/e2e/
#
# 2026-10-02 起兩版分開放（見 ../YOLO_baseline_端到端重跑規劃.md 3-1）：
#   e2e/  端到端（正式版），這支腳本上傳
#   nms/  一對多頭＋NMS（補充對照），已在 Drive 上，這支腳本只核對、不重傳
#
# 上傳內容：
#   e2e/seed{0,1,2}/video_inference/thr{0.1,0.3,0.5}/{train,valid,test}/*.mp4、video_classification.{csv,md}
#   e2e/thresholds_0.1_0.3_0.5/metrics_thr0.1_0.3_0.5.md   門檻 0.1、0.3、0.5 的數字表
#   README.md（頂層）                                          本資料夾 README 的副本（影片怎麼看、門檻定義）
# rclone copy 會跳過雲端已有且相同的檔案，所以重跑只補傳新的。
# 不上傳：權重檔、預測檔、log。
#
# 用法：nohup bash upload_videos_to_gdrive.sh > /path/to/data/baseline_yolo26l_v17/e2e/logs/upload_gdrive.log 2>&1 &
set -euo pipefail
RCLONE=~/.local/bin/rclone
ROOT=/path/to/data/baseline_yolo26l_v17
DROOT="gdrive:Co-DETR_training/baseline_yolo26l_v17"
HERE="$(cd "$(dirname "$0")" && pwd)"

# README 副本放在頂層（內容與 baseline/yolo/README.md 相同）
cp "$HERE/README.md" "$ROOT/README.md"

FILTER=$(mktemp)
cat > "$FILTER" <<'EOF'
+ /seed*/video_inference/thr*/**
+ /thresholds_0.1_0.3_0.5/metrics_thr0.1_0.3_0.5.md
- **
EOF
NFILTER=$(mktemp)
cat > "$NFILTER" <<'EOF'
+ /seed*/video_inference/thr*/**
+ /thresholds_0.3_0.5/metrics_thr0.3_0.5.md
+ /thresholds_0.1_0.3_0.5/metrics_thr0.1_0.3_0.5.md
- **
EOF

echo "=== 上傳開始 $(date '+%F %T') ==="
$RCLONE copyto "$ROOT/README.md" "$DROOT/README.md" -v
$RCLONE copy "$ROOT/e2e" "$DROOT/e2e" --filter-from "$FILTER" --transfers 4 --stats 30s --stats-one-line -v
echo "=== 上傳完成，開始逐檔比對 $(date '+%F %T') ==="
$RCLONE check "$ROOT/e2e" "$DROOT/e2e" --filter-from "$FILTER"
$RCLONE check "$ROOT/nms" "$DROOT/nms" --filter-from "$NFILTER"
L_MD5=$(md5sum "$ROOT/README.md" | cut -d' ' -f1); R_MD5=$($RCLONE md5sum "$DROOT/README.md" | cut -d' ' -f1)
[ "$L_MD5" = "$R_MD5" ] && echo "README.md 一致" || { echo "README.md 不一致：$L_MD5 vs $R_MD5"; exit 1; }
echo "UPLOAD_AND_CHECK_DONE $(date '+%F %T')"
rm -f "$FILTER" "$NFILTER"
