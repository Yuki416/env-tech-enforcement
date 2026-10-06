#!/usr/bin/env bash
# 第13次實驗（v14 動態版）訓練後全套分析。沿用第9次(v11) 的 6-part 流程與腳本，
# 只把路徑指到 v14 + baseline 指到第9次 v11。分析腳本(error_analysis_v9 / post_analysis_9th /
# visualize_litter_tp_comparison_9th) 皆為參數化、5類與 v14 相同，直接沿用。
#
# 使用方式：
#   nohup bash experiments/13th_v14_own_dynamic/post_analysis_13th_v14.sh \
#       > /path/to/data/13th_expt_v14_own_dynamic_36e/post_analysis.log 2>&1 &
#   tail -f /path/to/data/13th_expt_v14_own_dynamic_36e/post_analysis.log
#
# 重點：test = 原始 v11 靜止版 → Part1 的 test 誤差分析 + Part5 的對比，就是本實驗核心問題
#      「動態訓練能否泛化到靜止落地垃圾」的量化答案（對照第9次 v11 同 test）。

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # = co-detr 根目錄

CONTAINER="co-detr-v14-post"
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
RESULT_DIR="/workspace/hdd2_storage/13th_expt_v14_own_dynamic_36e"
CONFIG="/workspace/project/configs/co_dino_chiayi_v14_own_dynamic_36e.py"
CONFIG_NAME="co_dino_chiayi_v14_own_dynamic_36e"
DATA_ROOT="/workspace/datasets/coco/chiayi_v14_own_dynamic_litter"
# baseline = 第9次 v11（同 domain、完整靜+動標註，同一 static test）
BASELINE_DIR="/workspace/hdd2_storage/chiayi_v11_36e_65vids"
BASELINE_ERROR_DIR="${BASELINE_DIR}/error_analysis_score05"
BASELINE_CONFIG="/workspace/project/configs/co_dino_chiayi_v11_36e_65vids.py"
BASELINE_DATA_ROOT="/workspace/datasets/coco/chiayi_v11_garbage_litter_corrected"

# ── 報告標籤（傳給沿用的分析腳本，避免報告寫死成第9次名稱）────────────────────
EXP_LABEL="第13次 v14"
BASELINE_LABEL="第9次 v11"
COMPARISON_NAME="comparison_v11_9th_vs_v14_13th.md"
CONFIG_DIFF_MD="/workspace/project/experiments/13th_v14_own_dynamic/config_diff_v11_vs_v14.md"

# ── 自動偵測 v14 best checkpoint ──────────────────────────────────────────────
CHECKPOINT=$(docker exec "$CONTAINER" bash -c \
    "ls ${RESULT_DIR}/best_coco_litter_precision_epoch_*.pth 2>/dev/null | tail -1")
if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: 找不到 ${RESULT_DIR}/best_coco_litter_precision_epoch_*.pth（訓練是否完成？）"; exit 1
fi
echo "使用 v14 best checkpoint: ${CHECKPOINT}"

# ── Part 1a/1b: 誤差分析（score 0.5 / 0.3，train/valid/test）─────────────────
for THR in 0.5 0.3; do
    TAG=$(echo "$THR" | tr -d '.'); TAG="score${TAG}"
    echo "=== Part 1: Error analysis (score_thr=${THR}) → ${TAG} ==="
    docker exec -e CUDA_VISIBLE_DEVICES=0 "$CONTAINER" python /workspace/project/tools/error_analysis_v9.py \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" --data-root "$DATA_ROOT" \
        --output-dir "${RESULT_DIR}/error_analysis_${TAG}" \
        --splits train valid test --score-threshold "$THR"
done

# ── Part 2: AP 曲線 ───────────────────────────────────────────────────────────
echo "=== Part 2: AP 曲線 ==="
docker exec "$CONTAINER" python /workspace/project/tools/post_analysis_9th.py --mode ap-curve \
    --log "${RESULT_DIR}/training_log.md" --output-dir "${RESULT_DIR}/error_analysis_score05" \
    --exp-label "$EXP_LABEL" --config-name "$CONFIG_NAME"

# ── Part 3a/3b: 場景分組 ──────────────────────────────────────────────────────
for THR in 05 03; do
    echo "=== Part 3: 場景分組 (score${THR}) ==="
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

# ── Part 5: 第9次 v11(baseline) vs 第13次 v14 對比 ────────────────────────────
echo "=== Part 5: v11(9th) vs v14(13th) 對比 ==="
docker exec "$CONTAINER" python /workspace/project/tools/post_analysis_9th.py --mode comparison \
    --v7-log "${BASELINE_DIR}/training_log.md" --v8-log "${RESULT_DIR}/training_log.md" \
    --v7-error-dir "$BASELINE_ERROR_DIR" --v8-error-dir "${RESULT_DIR}/error_analysis_score05" \
    --output-dir "${RESULT_DIR}/error_analysis_score05" \
    --baseline-label "$BASELINE_LABEL" --exp-label "$EXP_LABEL" \
    --comparison-name "$COMPARISON_NAME" --config-diff-md "$CONFIG_DIFF_MD"

# ── Part 6: Litter/Garbage TP 視覺化比較（v11 vs v14）────────────────────────
echo "=== Part 6: TP 視覺化比較（v11 vs v14）==="
BASELINE_CKPT=$(docker exec "$CONTAINER" bash -c \
    "ls ${BASELINE_DIR}/best_coco_litter_precision_epoch_*.pth 2>/dev/null | tail -1")
if [ -z "$BASELINE_CKPT" ]; then
    echo "WARNING: 找不到 v11 baseline checkpoint，跳過 Part 6。"
else
    docker exec -e CUDA_VISIBLE_DEVICES=0 "$CONTAINER" \
        python /workspace/project/tools/visualize_litter_tp_comparison_9th.py \
        --v7-checkpoint "$BASELINE_CKPT" --v8-checkpoint "$CHECKPOINT" \
        --v7-ea-dir "$BASELINE_ERROR_DIR" --v8-ea-dir "${RESULT_DIR}/error_analysis_score05" \
        --v7-config "$BASELINE_CONFIG" --v7-data-root "$BASELINE_DATA_ROOT" --v7-label "9th_v11" \
        --v8-config "$CONFIG" --v8-data-root "$DATA_ROOT" --v8-label "13th_v14" \
        --output-dir "${RESULT_DIR}/litter_tp_comparison" \
        || echo "WARNING: Part 6 失敗，跳過（不影響前面結果）"
fi

echo ""; echo "=== Post-analysis 全部完成 ==="
echo "  誤差分析(0.5):  ${RESULT_DIR}/error_analysis_score05/"
echo "  誤差分析(0.3):  ${RESULT_DIR}/error_analysis_score03/"
echo "  AP 曲線:        ${RESULT_DIR}/error_analysis_score05/litter_garbage_ap_curve.md"
echo "  v11 vs v14 對比: ${RESULT_DIR}/error_analysis_score05/（comparison 輸出）"
echo "  TP 視覺化:      ${RESULT_DIR}/litter_tp_comparison/"
