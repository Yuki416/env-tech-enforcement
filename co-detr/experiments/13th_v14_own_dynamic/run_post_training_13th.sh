#!/usr/bin/env bash
# 等第13次訓練完成 → 自動接續：post-analysis → 影片推論。
# 訓練完成偵測 = 出現 epoch_36.pth（最後一個 epoch 存檔）。
#
# 建議用法（背景、可斷線）：
#   nohup bash experiments/13th_v14_own_dynamic/run_post_training_13th.sh \
#       > /path/to/data/13th_expt_v14_own_dynamic_36e/post_training_chain.log 2>&1 &
#   tail -f /path/to/data/13th_expt_v14_own_dynamic_36e/post_training_chain.log

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_HOST=/path/to/data/13th_expt_v14_own_dynamic_36e

# ── 1. 等訓練完成 ─────────────────────────────────────────────────────────────
echo "[$(date +%H:%M)] 等待訓練完成（偵測 ${RESULT_HOST}/epoch_36.pth）..."
TIMEOUT=$((8*3600)); WAITED=0
until [ -f "${RESULT_HOST}/epoch_36.pth" ]; do
    sleep 300; WAITED=$((WAITED+300))
    if [ "$WAITED" -ge "$TIMEOUT" ]; then
        echo "[$(date +%H:%M)] 逾時 8h 仍未見 epoch_36.pth，放棄（訓練是否異常？）"; exit 1
    fi
done
echo "[$(date +%H:%M)] 偵測到訓練完成。等 60s 確保 best ckpt 寫完。"; sleep 60
echo "  best ckpt: $(ls ${RESULT_HOST}/best_coco_litter_precision_epoch_*.pth 2>/dev/null | tail -1)"

# ── 2. Post-analysis（誤差分析/AP/場景/score/對比v11/TP視覺化）─────────────────
echo "[$(date +%H:%M)] === Post-analysis 開始 ==="
if bash "${SCRIPT_DIR}/post_analysis_13th_v14.sh"; then
    echo "[$(date +%H:%M)] Post-analysis 完成 ✓"
else
    echo "[$(date +%H:%M)] Post-analysis 有錯（exit $?），仍繼續影片推論。"
fi

# ── 3. 影片推論（train/valid/test，帶 bbox + TP/FP/FN 分類）────────────────────
echo "[$(date +%H:%M)] === 影片推論開始 ==="
docker run --rm --gpus all --ipc=host --shm-size=16g \
    -v "${CO_DIR}:/workspace/project" \
    -v /path/to/data/coco:/workspace/datasets/coco \
    -v /path/to/data:/workspace/hdd2_storage \
    -e PYTHONPATH=/opt/mmdetection:/workspace/project \
    -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 -e PYTHONUTF8=1 \
    co-detr:latest \
    python /workspace/project/experiments/13th_v14_own_dynamic/infer_13th_v14_video_classification.py \
        --splits test valid train \
    && echo "[$(date +%H:%M)] 影片推論完成 ✓" || echo "[$(date +%H:%M)] 影片推論有錯。"

# ── 4. 影片分類 IoU≥0.5 校正（video-inference Step 2）──────────────────────────
# 步驟 3 的分類只看「有沒有預測 litter/garbage」→ 隨便吐一個框就算 TP，會高估。
# 這裡用 error_analysis_score03 的 IoU≥0.5 匹配結果重算，並重渲染被改判影片的 HUD。
# （2026-07 補：先前此步驟被遺漏，導致 valid/test 各有 3/1 部影片被誤標為 TP。）
echo "[$(date +%H:%M)] === 影片分類 IoU 校正開始 ==="
docker run --rm \
    -v "${CO_DIR}:/workspace/project" \
    -v /path/to/data:/workspace/hdd2_storage \
    -e PYTHONUTF8=1 -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 \
    co-detr:latest \
    python /workspace/project/experiments/13th_v14_own_dynamic/fix_video_classifications_13th.py \
    && echo "[$(date +%H:%M)] IoU 校正完成 ✓" || echo "[$(date +%H:%M)] IoU 校正有錯。"

# 容器產出的檔案屬 root → 修正權限，否則後續 rclone 上傳/瀏覽會卡
docker run --rm -v /path/to/data:/path/to/data co-detr:latest \
    chown -R "$(id -u):$(id -g)" "${RESULT_HOST}/video_inference/" 2>/dev/null || true

echo "[$(date +%H:%M)] === 第13次訓練後處理全部結束 ==="
echo "結果都在: ${RESULT_HOST}/"
echo "（影片分類已含 IoU≥0.5 校正；上傳 Drive 見影片推論流程說明）"
