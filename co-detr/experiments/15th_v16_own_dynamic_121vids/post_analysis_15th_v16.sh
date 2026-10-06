#!/usr/bin/env bash
# 第15次實驗（v16 動態子集 / 121 部）訓練後全套分析。
# 沿用第13次的 6-part 流程，兩處關鍵改變：
#   1. baseline 從「第9次 v11」改成 **第13次 v14**（本次的對照組）
#   2. test 誤差分析跑 **兩份**：
#        test_static594（★主判定，與第13次同一把尺）→ error_analysis_score05 / score03
#        test 完整 655（參考，⚠️含同片洩漏）        → error_analysis_full655_score05
#      靠 error_analysis_v9.py 新增的 --split-ann 覆寫達成（該參數不給則行為與過去完全相同）。
#
# 使用方式：
#   nohup bash experiments/15th_v16_own_dynamic_121vids/post_analysis_15th_v16.sh \
#       > /path/to/data/15th_expt_v16_own_dynamic_121vids_36e/post_analysis.log 2>&1 &
#   tail -f /path/to/data/15th_expt_v16_own_dynamic_121vids_36e/post_analysis.log
#
# 核心問題：資料規模 44→99 clips（第13次訓練集是本次的嚴格子集）能否改善 litter 偵測？
#          答案看 Part 1 的 **test_static594** + Part 5 的對比。
# 日/夜：Part 3 by-scene 會輸出 Day_Litter/Night_Litter/Day_Garbage/Night_Garbage。
#        ⚠️ test 的 Night_Litter clip **litter 框 = 0**（自第9次即如此）→ 夜間 litter 只能看 **valid**
#           （valid Night_Litter 3 clips / 108 框）。詳見實驗夾 README「夜間訊號只能從 valid 取得」。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # = co-detr 根目錄

CONTAINER="co-detr-v16-post"
# 前一次執行若中途失敗，同名 container 會殘留 → docker run --name 直接失敗。先清掉再開。
docker rm -f "$CONTAINER" 2>/dev/null || true
docker run -d --name "$CONTAINER" \
    --gpus all --ipc=host --shm-size=16g \
    -v "${SCRIPT_DIR}:/workspace/project" \
    -v /path/to/data/coco:/workspace/datasets/coco \
    -v /path/to/data:/workspace/hdd2_storage \
    -e PYTHONPATH=/opt/mmdetection:/workspace/project \
    -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 -e PYTHONUTF8=1 \
    co-detr:latest sleep infinity
trap "echo 'Stopping container...'; docker rm -f $CONTAINER 2>/dev/null || true" EXIT

# ── 路徑 ──────────────────────────────────────────────────────────────────────
RESULT_DIR="/workspace/hdd2_storage/15th_expt_v16_own_dynamic_121vids_36e"
CONFIG="/workspace/project/configs/co_dino_chiayi_v16_own_dynamic_36e.py"
CONFIG_NAME="co_dino_chiayi_v16_own_dynamic_36e"
DATA_ROOT="/workspace/datasets/coco/chiayi_v16_own_dynamic_litter"
# baseline = 第13次 v14（同 config、同配方、train 集為本次的嚴格子集、同一 594 test）
BASELINE_DIR="/workspace/hdd2_storage/13th_expt_v14_own_dynamic_36e"
BASELINE_ERROR_DIR="${BASELINE_DIR}/error_analysis_score05"
BASELINE_CONFIG="/workspace/project/configs/co_dino_chiayi_v14_own_dynamic_36e.py"
BASELINE_DATA_ROOT="/workspace/datasets/coco/chiayi_v14_own_dynamic_litter"

# ── 報告標籤 ──────────────────────────────────────────────────────────────────
EXP_LABEL="第15次 v16"
BASELINE_LABEL="第13次 v14"
COMPARISON_NAME="comparison_v14_13th_vs_v16_15th.md"
CONFIG_DIFF_MD="/workspace/project/experiments/15th_v16_own_dynamic_121vids/config_diff_v14_vs_v16.md"

# ── 自動偵測 best checkpoint ──────────────────────────────────────────────────
CHECKPOINT=$(docker exec "$CONTAINER" bash -c \
    "ls ${RESULT_DIR}/best_coco_litter_precision_epoch_*.pth 2>/dev/null | tail -1")
if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: 找不到 ${RESULT_DIR}/best_coco_litter_precision_epoch_*.pth（訓練是否完成？）"; exit 1
fi
echo "使用 v16 best checkpoint: ${CHECKPOINT}"

# ── Part 1a/1b: 誤差分析（score 0.5 / 0.3，train/valid/test）─────────────────
#    ★ test 用 test_static594.json（主判定），與第13次同一把尺
for THR in 0.5 0.3; do
    TAG=$(echo "$THR" | tr -d '.'); TAG="score${TAG}"
    echo "=== Part 1: Error analysis (score_thr=${THR}) → ${TAG}  [test = 594 靜止子集] ==="
    docker exec -e CUDA_VISIBLE_DEVICES=0 "$CONTAINER" python /workspace/project/tools/error_analysis_v9.py \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" --data-root "$DATA_ROOT" \
        --output-dir "${RESULT_DIR}/error_analysis_${TAG}" \
        --splits train valid test --score-threshold "$THR" \
        --split-ann test=annotations/test_static594.json
done

# ── Part 1c: 完整 655 test（score 0.5 / 0.3）─────────────────────────────────
#   score05：參考數字（⚠️ 含同片洩漏，不可用來宣稱進步）
#   score03：**IoU 校正必需** —— 影片推論讀 test.json（655/8 部），
#            而 error_analysis_score03 的 test 只有 594/7 部，
#            若用它校正，v16 新增的第 8 部會查無資料 → fix 腳本會中止（已加涵蓋率檢查）。
for THR in 0.5 0.3; do
    TAG=$(echo "$THR" | tr -d '.'); TAG="score${TAG}"
    echo "=== Part 1c: Error analysis — 完整 655 test (score_thr=${THR}) ==="
    docker exec -e CUDA_VISIBLE_DEVICES=0 "$CONTAINER" python /workspace/project/tools/error_analysis_v9.py \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" --data-root "$DATA_ROOT" \
        --output-dir "${RESULT_DIR}/error_analysis_full655_${TAG}" \
        --splits test --score-threshold "$THR" \
        || echo "WARNING: Part 1c (${TAG}) 失敗"
done

# ── Part 2: AP 曲線 ───────────────────────────────────────────────────────────
echo "=== Part 2: AP 曲線 ==="
docker exec "$CONTAINER" python /workspace/project/tools/post_analysis_9th.py --mode ap-curve \
    --log "${RESULT_DIR}/training_log.md" --output-dir "${RESULT_DIR}/error_analysis_score05" \
    --exp-label "$EXP_LABEL" --config-name "$CONFIG_NAME"

# ── Part 3a/3b: 場景分組（日/夜）──────────────────────────────────────────────
#    ⚠️ 夜間 litter 只有 valid 有樣本（test 的 Night_Litter clip 無 litter 標註）
for THR in 05 03; do
    echo "=== Part 3: 場景分組 日/夜 (score${THR}) ==="
    docker exec "$CONTAINER" python /workspace/project/tools/post_analysis_9th.py --mode by-scene \
        --error-dir "${RESULT_DIR}/error_analysis_score${THR}" \
        --output-dir "${RESULT_DIR}/error_analysis_score${THR}" \
        --output-name "litter_garbage_by_scene_score${THR}.md" \
        --exp-label "$EXP_LABEL"
done

# ── Part 4: Score 分布 ────────────────────────────────────────────────────────
echo "=== Part 4: Score 分布 ==="
docker exec -e CUDA_VISIBLE_DEVICES=0 "$CONTAINER" python /workspace/project/tools/post_analysis_9th.py \
    --mode score-dist --config "$CONFIG" --checkpoint "$CHECKPOINT" --data-root "$DATA_ROOT" \
    --output-dir "${RESULT_DIR}/error_analysis_score05" \
    --exp-label "$EXP_LABEL"

# ── Part 5: 第13次 v14(baseline) vs 第15次 v16 對比 ★核心 ────────────────────
echo "=== Part 5: v14(13th) vs v16(15th) 對比 ==="
docker exec "$CONTAINER" python /workspace/project/tools/post_analysis_9th.py --mode comparison \
    --v7-log "${BASELINE_DIR}/training_log.md" --v8-log "${RESULT_DIR}/training_log.md" \
    --v7-error-dir "$BASELINE_ERROR_DIR" --v8-error-dir "${RESULT_DIR}/error_analysis_score05" \
    --output-dir "${RESULT_DIR}/error_analysis_score05" \
    --baseline-label "$BASELINE_LABEL" --exp-label "$EXP_LABEL" \
    --comparison-name "$COMPARISON_NAME" --config-diff-md "$CONFIG_DIFF_MD"

# ── Part 6: Litter/Garbage TP 視覺化比較（v14 vs v16）────────────────────────
#    ⚠️ 第13次修好的參數化版本：config/data_root 皆由參數傳入（舊版寫死 v10/v11 → 產物無效）
echo "=== Part 6: TP 視覺化比較（v14 vs v16）==="
BASELINE_CKPT=$(docker exec "$CONTAINER" bash -c \
    "ls ${BASELINE_DIR}/best_coco_litter_precision_epoch_*.pth 2>/dev/null | tail -1")
if [ -z "$BASELINE_CKPT" ]; then
    echo "WARNING: 找不到第13次 baseline checkpoint，跳過 Part 6。"
else
    docker exec -e CUDA_VISIBLE_DEVICES=0 "$CONTAINER" \
        python /workspace/project/tools/visualize_litter_tp_comparison_9th.py \
        --v7-checkpoint "$BASELINE_CKPT" --v8-checkpoint "$CHECKPOINT" \
        --v7-ea-dir "$BASELINE_ERROR_DIR" --v8-ea-dir "${RESULT_DIR}/error_analysis_score05" \
        --v7-config "$BASELINE_CONFIG" --v7-data-root "$BASELINE_DATA_ROOT" --v7-label "13th_v14" \
        --v8-config "$CONFIG" --v8-data-root "$DATA_ROOT" --v8-label "15th_v16" \
        --output-dir "${RESULT_DIR}/litter_tp_comparison" \
        || echo "WARNING: Part 6 失敗，跳過（不影響前面結果）"
fi

echo ""; echo "=== Post-analysis 全部完成 ==="
echo "  ★主判定(594) 誤差分析(0.5): ${RESULT_DIR}/error_analysis_score05/"
echo "   主判定(594) 誤差分析(0.3): ${RESULT_DIR}/error_analysis_score03/"
echo "   參考(655,含洩漏):          ${RESULT_DIR}/error_analysis_full655_score05/"
echo "   日/夜分組:                 ${RESULT_DIR}/error_analysis_score05/litter_garbage_by_scene_score05.md"
echo "   v14 vs v16 對比:           ${RESULT_DIR}/error_analysis_score05/${COMPARISON_NAME}"
echo "   TP 視覺化:                 ${RESULT_DIR}/litter_tp_comparison/"
