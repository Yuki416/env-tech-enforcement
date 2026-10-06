#!/usr/bin/env bash
# 第15次訓練完成後 → 自動接續全套後處理。
# 訓練完成偵測 = 出現 epoch_36.pth（最後一個 epoch 存檔）。
#
# 建議用法（背景、可斷線）：
#   nohup bash experiments/15th_v16_own_dynamic_121vids/run_post_training_15th.sh \
#       > /path/to/data/15th_expt_v16_own_dynamic_121vids_36e/post_training_chain.log 2>&1 &
#   tail -f /path/to/data/15th_expt_v16_own_dynamic_121vids_36e/post_training_chain.log
#
# 步驟：1 等訓練完 → 2 ★主判定評估(594) → 3 post-analysis 6-part → 4 影片推論
#       → 5 IoU≥0.5 校正 → 6 AP 曲線正式版 → 7 權限修正
# ⚠️ 第 13、14 次都曾漏掉步驟 5（IoU 校正），本腳本已內建，不會再漏。

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULT_HOST=/path/to/data/15th_expt_v16_own_dynamic_121vids_36e

# ── 1. 等訓練完成 ─────────────────────────────────────────────────────────────
echo "[$(date +%H:%M)] 等待訓練完成（偵測 ${RESULT_HOST}/epoch_36.pth）..."
TIMEOUT=$((16*3600)); WAITED=0     # 本次資料 2.43×，預估 ~14h，逾時放寬到 16h
until [ -f "${RESULT_HOST}/epoch_36.pth" ]; do
    sleep 300; WAITED=$((WAITED+300))
    if [ "$WAITED" -ge "$TIMEOUT" ]; then
        echo "[$(date +%H:%M)] 逾時 16h 仍未見 epoch_36.pth，放棄（訓練是否異常？）"; exit 1
    fi
done
echo "[$(date +%H:%M)] 偵測到 epoch_36.pth。"

# ⚠️ epoch_36.pth 是在 after_train_epoch 存的，**最後一輪 val 還沒跑**（MMEngine 先存檔再 val）。
#    那一輪 val 仍佔著 GPU（~18GB），且 save_best 可能在 val 後才更新 best ckpt。
#    第 13 次只 sleep 60s 剛好蓋過（實測最後一輪 val 僅 28s），但那是運氣不是設計：
#    若太早開始 → GPU 爭用可能 OOM，且可能抓到**過期的 best ckpt**。
#    這裡改成等訓練 container 真正結束，穩固且零成本。
echo "[$(date +%H:%M)] 等訓練 container 結束（最後一輪 val + save_best 尚未完成）..."
WAITED=0
while docker ps --no-trunc --format '{{.Command}}' 2>/dev/null | grep -q "dist_train"; do
    sleep 30; WAITED=$((WAITED+30))
    if [ "$WAITED" -ge 1800 ]; then
        echo "[$(date +%H:%M)] 警告：等了 30 分鐘 container 仍在，繼續往下（請自行確認）"; break
    fi
done
echo "[$(date +%H:%M)] 訓練 container 已結束（等待 ${WAITED}s）。再等 30s 確保檔案落盤。"; sleep 30

BEST=$(ls ${RESULT_HOST}/best_coco_litter_precision_epoch_*.pth 2>/dev/null | tail -1)
echo "  best ckpt: ${BEST}"
if [ -z "$BEST" ]; then echo "ERROR: 找不到 best ckpt，中止。"; exit 1; fi
# save_best 每次只保留一個 best 檔；若出現多個代表狀態異常，先擋下來
N_BEST=$(ls ${RESULT_HOST}/best_coco_litter_precision_epoch_*.pth 2>/dev/null | wc -l)
if [ "$N_BEST" -ne 1 ]; then
    echo "ERROR: best ckpt 有 ${N_BEST} 個（正常應為 1），恐抓錯 checkpoint，中止。"
    ls ${RESULT_HOST}/best_coco_litter_precision_epoch_*.pth; exit 1
fi
# GPU 應已淨空，否則後續推論可能 OOM
echo "  GPU 狀態: $(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | tr '\n' ' ')"

# ── 2. ★主判定：594 靜止子集評估（15th vs 13th 同一把尺）─────────────────────
echo "[$(date +%H:%M)] === ★主判定：594 靜止子集評估 開始 ==="
if bash "${SCRIPT_DIR}/eval_15th_vs_13th_on_static594.sh"; then
    echo "[$(date +%H:%M)] 主判定評估完成 ✓"
else
    echo "[$(date +%H:%M)] 主判定評估有錯（exit $?），仍繼續。"
fi

# ── 3. Post-analysis 6-part ───────────────────────────────────────────────────
echo "[$(date +%H:%M)] === Post-analysis 開始 ==="
if bash "${SCRIPT_DIR}/post_analysis_15th_v16.sh"; then
    echo "[$(date +%H:%M)] Post-analysis 完成 ✓"
else
    echo "[$(date +%H:%M)] Post-analysis 有錯（exit $?），仍繼續影片推論。"
fi

# ── 4. 影片推論（train/valid/test，帶 bbox + TP/FP/FN 分類）────────────────────
echo "[$(date +%H:%M)] === 影片推論開始 ==="
docker run --rm --gpus all --ipc=host --shm-size=16g \
    -v "${CO_DIR}:/workspace/project" \
    -v /path/to/data/coco:/workspace/datasets/coco \
    -v /path/to/data:/workspace/hdd2_storage \
    -e PYTHONPATH=/opt/mmdetection:/workspace/project \
    -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 -e PYTHONUTF8=1 \
    co-detr:latest \
    python /workspace/project/experiments/15th_v16_own_dynamic_121vids/infer_15th_v16_video_classification.py \
        --splits test valid train \
    && echo "[$(date +%H:%M)] 影片推論完成 ✓" || echo "[$(date +%H:%M)] 影片推論有錯。"

# ── 5. 影片分類 IoU≥0.5 校正（video-inference Step 2）──────────────────────────
# 步驟 4 的分類只看「有沒有預測 litter/garbage」→ 位置全錯的框也算 TP，會高估。
# 這裡用 error_analysis_score03 的 IoU≥0.5 匹配結果重算，並重渲染被改判影片的 HUD。
# ⚠️ 第 13、14 次都曾漏掉這步（第13次補做後實測 4 部影片由 TP 改判 FN）。
echo "[$(date +%H:%M)] === 影片分類 IoU≥0.5 校正開始 ==="
docker run --rm \
    -v "${CO_DIR}:/workspace/project" \
    -v /path/to/data:/workspace/hdd2_storage \
    -e PYTHONUTF8=1 -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 \
    co-detr:latest \
    python /workspace/project/experiments/15th_v16_own_dynamic_121vids/fix_video_classifications_15th.py \
    && echo "[$(date +%H:%M)] IoU 校正完成 ✓" || echo "[$(date +%H:%M)] IoU 校正有錯。"

# ── 6. AP vs epoch 正式版（36 epoch 全數到齊後重畫，覆蓋訓練中的暫定版）────────
echo "[$(date +%H:%M)] === AP 曲線正式版 ==="
docker run --rm \
    -v "${CO_DIR}:/workspace/project" \
    -v /path/to/data:/path/to/data \
    -e PYTHONUTF8=1 \
    co-detr:latest \
    python /workspace/project/experiments/15th_v16_own_dynamic_121vids/plot_ap_vs_epoch_15th.py \
    && echo "[$(date +%H:%M)] AP 曲線完成 ✓" || echo "[$(date +%H:%M)] AP 曲線有錯。"

# ── 7. 容器產出屬 root → 修正權限，否則後續 rclone 上傳/瀏覽會卡 ───────────────
docker run --rm -v /path/to/data:/path/to/data co-detr:latest \
    chown -R "$(id -u):$(id -g)" "${RESULT_HOST}/" 2>/dev/null || true

echo "[$(date +%H:%M)] === 第15次訓練後處理全部結束 ==="
echo "結果都在: ${RESULT_HOST}/"
echo ""
echo "★ 先看這個（主判定，與第13次同一把尺）："
echo "   ${RESULT_HOST}/eval_static594_vs_13th/"
echo "   ${RESULT_HOST}/error_analysis_score05/   ← 逐框 TP/FN/FP（test=594 子集）"
echo "   ${RESULT_HOST}/error_analysis_score05/comparison_v14_13th_vs_v16_15th.md"
echo ""
echo "⚠️ error_analysis_full655_score05/ 含已知同片洩漏 → 僅參考，不可用來宣稱進步"
echo "⚠️ 夜間 litter 請看 valid（test 的 Night_Litter clip 無 litter 標註）"
echo "（影片分類已含 IoU≥0.5 校正；上傳 Drive 見影片推論流程說明）"
