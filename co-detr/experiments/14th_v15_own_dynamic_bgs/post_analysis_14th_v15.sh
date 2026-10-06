#!/usr/bin/env bash
# 第14次實驗（v15 = 動態子集 + BGS changemap 4ch）訓練後全套分析。
# 沿用第 9/13 次的 6-part 流程與 tools/ 下的參數化腳本，只把路徑指到 v15 + baseline 指到第 13 次 v14。
#
# ⚠️ 4ch 專屬風險（本次特別防範）：
#   hooks/load_changemap.py 的 LoadImageAndChangemap 找不到 changemap 檔時會**靜默補全零通道、不報錯**。
#   若 DATA_ROOT 誤指到 v14（無 changemap），整套分析會「跑完但數字全錯」。故啟動前強制檢查。
#   Part 1/4/6 用 init_detector + inference_detector（吃 config 的 test_pipeline）→ 已驗證 4ch 相容
#   （見 verify_4ch_inference.py）。Part 2/3/5 純讀 log/檔案，不推論。
#
# 使用方式：
#   nohup bash experiments/14th_v15_own_dynamic_bgs/post_analysis_14th_v15.sh \
#       > /path/to/data/14th_expt_v15_own_dynamic_bgs_4ch_36e/post_analysis.log 2>&1 &
#   tail -f /path/to/data/14th_expt_v15_own_dynamic_bgs_4ch_36e/post_analysis.log

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # = co-detr 根目錄

# ── 路徑（host 端先檢查，再進容器）────────────────────────────────────────────
HOST_V15=/path/to/data/coco/chiayi_v15_own_dynamic_bgs
for s in train valid test; do
    if [ ! -d "${HOST_V15}/${s}_changemap" ]; then
        echo "❌ 找不到 ${HOST_V15}/${s}_changemap —— 4ch 會靜默補零、結果全錯。中止。"; exit 1
    fi
done
echo "✅ changemap 目錄檢查通過（train/valid/test 皆有）"

CONTAINER="co-detr-v15-post"
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

RESULT_DIR="/workspace/hdd2_storage/14th_expt_v15_own_dynamic_bgs_4ch_36e"
CONFIG="/workspace/project/configs/co_dino_chiayi_v15_own_dynamic_bgs_4ch_36e.py"
CONFIG_NAME="co_dino_chiayi_v15_own_dynamic_bgs_4ch_36e"
DATA_ROOT="/workspace/datasets/coco/chiayi_v15_own_dynamic_bgs"      # ★ v15（有 changemap）

# baseline = 第13次 v14（同影像同標註、純 RGB，同一 static test）
BASELINE_DIR="/workspace/hdd2_storage/13th_expt_v14_own_dynamic_36e"
BASELINE_ERROR_DIR="${BASELINE_DIR}/error_analysis_score05"
BASELINE_CONFIG="/workspace/project/configs/co_dino_chiayi_v14_own_dynamic_36e.py"
BASELINE_DATA_ROOT="/workspace/datasets/coco/chiayi_v14_own_dynamic_litter"

EXP_LABEL="第14次 v15 BGS4ch"
BASELINE_LABEL="第13次 v14 RGB"
COMPARISON_NAME="comparison_v14_13th_vs_v15_14th.md"
CONFIG_DIFF_MD="/workspace/project/experiments/14th_v15_own_dynamic_bgs/config_diff_v14_vs_v15.md"

# ── 自動偵測 v15 best checkpoint ──────────────────────────────────────────────
CHECKPOINT=$(docker exec "$CONTAINER" bash -c \
    "ls ${RESULT_DIR}/best_coco_litter_precision_epoch_*.pth 2>/dev/null | tail -1")
if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: 找不到 ${RESULT_DIR}/best_coco_litter_precision_epoch_*.pth"; exit 1
fi
echo "使用 v15 best checkpoint: ${CHECKPOINT}"

BASELINE_CKPT=$(docker exec "$CONTAINER" bash -c \
    "ls ${BASELINE_DIR}/best_coco_litter_precision_epoch_*.pth 2>/dev/null | tail -1")
echo "baseline (第13次) checkpoint: ${BASELINE_CKPT:-<未找到>}"

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

# ── Part 5: 第13次 v14(baseline) vs 第14次 v15 對比 ───────────────────────────
echo "=== Part 5: v14(13th) vs v15(14th) 對比 ==="
docker exec "$CONTAINER" python /workspace/project/tools/post_analysis_9th.py --mode comparison \
    --v7-log "${BASELINE_DIR}/training_log.md" --v8-log "${RESULT_DIR}/training_log.md" \
    --v7-error-dir "$BASELINE_ERROR_DIR" --v8-error-dir "${RESULT_DIR}/error_analysis_score05" \
    --output-dir "${RESULT_DIR}/error_analysis_score05" \
    --baseline-label "$BASELINE_LABEL" --exp-label "$EXP_LABEL" \
    --comparison-name "$COMPARISON_NAME" --config-diff-md "$CONFIG_DIFF_MD"

# ── Part 6: Litter/Garbage TP 視覺化比較（第13次 vs 第14次）──────────────────
echo "=== Part 6: TP 視覺化比較（13th v14 vs 14th v15）==="
if [ -z "$BASELINE_CKPT" ]; then
    echo "WARNING: 找不到第13次 baseline checkpoint，跳過 Part 6。"
else
    docker exec -e CUDA_VISIBLE_DEVICES=0 "$CONTAINER" \
        python /workspace/project/tools/visualize_litter_tp_comparison_9th.py \
        --v7-checkpoint "$BASELINE_CKPT" --v8-checkpoint "$CHECKPOINT" \
        --v7-ea-dir "$BASELINE_ERROR_DIR" --v8-ea-dir "${RESULT_DIR}/error_analysis_score05" \
        --v7-config "$BASELINE_CONFIG" --v7-data-root "$BASELINE_DATA_ROOT" --v7-label "13th_v14" \
        --v8-config "$CONFIG" --v8-data-root "$DATA_ROOT" --v8-label "14th_v15" \
        --output-dir "${RESULT_DIR}/litter_tp_comparison" \
        || echo "WARNING: Part 6 失敗，跳過（不影響前面結果）"
fi

echo ""; echo "=== Post-analysis 全部完成 ==="
echo "  誤差分析(0.5):   ${RESULT_DIR}/error_analysis_score05/"
echo "  誤差分析(0.3):   ${RESULT_DIR}/error_analysis_score03/"
echo "  AP 曲線:         ${RESULT_DIR}/error_analysis_score05/litter_garbage_ap_curve.md"
echo "  13th vs 14th 對比: ${RESULT_DIR}/error_analysis_score05/${COMPARISON_NAME}"
echo "  TP 視覺化:       ${RESULT_DIR}/litter_tp_comparison/"
echo "POST_ANALYSIS_DONE"
