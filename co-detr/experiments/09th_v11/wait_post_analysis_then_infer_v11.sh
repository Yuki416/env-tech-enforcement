#!/usr/bin/env bash
# wait_post_analysis_then_infer_v11.sh
#
# 等待 post-analysis 完成後，自動執行 v11 影片推論兩步驟：
#   Step 1 — infer_v11_video_classification.py（產生 MP4 + 初步 CSV）
#   Step 2 — fix_video_classifications_v11.py（IoU 嚴格比對修正分類）
#
# 使用方式（背景執行）：
#   nohup bash wait_post_analysis_then_infer_v11.sh > results/chiayi_v11_36e_65vids/video_inference.log 2>&1 &
#   tail -f results/chiayi_v11_36e_65vids/video_inference.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POST_LOG="${SCRIPT_DIR}/results/chiayi_v11_36e_65vids/post_analysis.log"
EA_DIR="${SCRIPT_DIR}/results/chiayi_v11_36e_65vids/error_analysis_score03"
HDD_OUT="/path/to/data/chiayi_v11_36e_65vids"
CHECK_INTERVAL=60

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ── Phase 1: 等待 post-analysis 完成 ─────────────────────────────────────────
log "========================================================="
log " V11 影片推論：等待 post-analysis 完成"
log "========================================================="
log ""
log "Phase 1: 監控 post-analysis..."

while true; do
    if grep -q "=== 全部完成 ===" "$POST_LOG" 2>/dev/null; then
        log "  post-analysis 已完成 ✓"
        break
    fi
    LAST=$(tail -1 "$POST_LOG" 2>/dev/null || echo "（尚無輸出）")
    log "  等待中... 最新進度: $LAST"
    sleep "$CHECK_INTERVAL"
done

# 確認 error_analysis_score03 存在
if [ ! -d "$EA_DIR" ]; then
    log "ERROR: 找不到 ${EA_DIR}，post-analysis 可能未正常完成"
    exit 1
fi
log "  error_analysis_score03 確認存在 ✓"

# ── Phase 2: 建立 HDD 輸出目錄 ───────────────────────────────────────────────
log ""
log "Phase 2: 建立 HDD 輸出目錄"
mkdir -p "${HDD_OUT}/video_inference"
log "  ${HDD_OUT}/video_inference ✓"

# ── Phase 3: 影片推論（產生 MP4 + 初步 CSV）─────────────────────────────────
log ""
log "Phase 3: 啟動影片推論（infer_v11_video_classification.py）"
log "  輸出目錄: ${HDD_OUT}/video_inference/"

docker run --rm \
    --name co-detr-v11-infer \
    --gpus '"device=0"' \
    --ipc host \
    --shm-size 8g \
    -v "${SCRIPT_DIR}:/workspace/project" \
    -v /path/to/data:/path/to/data \
    -e PYTHONPATH=/opt/mmdetection:/workspace/project \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e LANG=C.UTF-8 \
    -e LC_ALL=C.UTF-8 \
    co-detr:latest \
    python /workspace/project/infer_v11_video_classification.py \
        --splits train valid test \
        --device cuda:0

log "  影片推論完成 ✓"

# ── Phase 4: 修正分類（IoU ≥ 0.5 嚴格比對）─────────────────────────────────
log ""
log "Phase 4: 修正分類（fix_video_classifications_v11.py）"

docker run --rm \
    --name co-detr-v11-fix \
    -v "${SCRIPT_DIR}:/workspace/project" \
    -v /path/to/data:/path/to/data \
    -e LANG=C.UTF-8 \
    -e LC_ALL=C.UTF-8 \
    co-detr:latest \
    python /workspace/project/fix_video_classifications_v11.py

log "  分類修正完成 ✓"

# ── Phase 5: 修正 Docker 產生的 root 權限 ────────────────────────────────────
log ""
log "Phase 5: 修正檔案權限"

docker run --rm \
    -v /path/to/data:/path/to/data \
    alpine:latest \
    chmod -R 644 /path/to/data/chiayi_v11_36e_65vids/video_inference/

log "  權限修正完成 ✓"

# ── 完成 ──────────────────────────────────────────────────────────────────────
log ""
log "========================================================="
log " 全部完成！"
log "  MP4 影片: ${HDD_OUT}/video_inference/{train,valid,test}/"
log "  分類結果: ${HDD_OUT}/video_inference/video_classification.csv"
log "  摘要報告: ${HDD_OUT}/video_inference/video_classification_summary.md"
log "========================================================="
