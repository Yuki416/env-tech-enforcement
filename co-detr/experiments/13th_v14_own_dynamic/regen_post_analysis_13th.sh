#!/usr/bin/env bash
# 一次性：用「就地參數化」後的分析腳本，重新產生第13次報告的正確命名版本。
#   - Part 1（誤差分析）只有標題寫死錯誤 → 就地 sed 改標題（等價重跑、零 GPU）。
#   - Part 2/3/5（CPU，讀 CSV/log）+ Part 4/6（GPU 推論）→ 用正確 label/config/data 重跑。
#   - 清掉舊的錯名輸出（v11_8th_vs_9th_comparison.md、litter_tp_comparison/{8th,9th}）。
#
# 用法（背景）：
#   nohup bash experiments/13th_v14_own_dynamic/regen_post_analysis_13th.sh \
#     > /path/to/data/13th_expt_v14_own_dynamic_36e/regen_post_analysis.log 2>&1 &
#   tail -f /path/to/data/13th_expt_v14_own_dynamic_36e/regen_post_analysis.log

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # = co-detr 根目錄

CONTAINER="co-detr-v14-regen"
docker run -d --name "$CONTAINER" \
    --gpus all --ipc=host --shm-size=16g \
    -v "${SCRIPT_DIR}:/workspace/project" \
    -v /path/to/data/coco:/workspace/datasets/coco \
    -v /path/to/data:/workspace/hdd2_storage \
    -e PYTHONPATH=/opt/mmdetection:/workspace/project \
    -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 -e PYTHONUTF8=1 \
    co-detr:latest sleep infinity
trap "echo 'Stopping container...'; docker rm -f $CONTAINER 2>/dev/null || true" EXIT

# ── 路徑（與 post_analysis_13th_v14.sh 一致）─────────────────────────────────
RESULT_DIR="/workspace/hdd2_storage/13th_expt_v14_own_dynamic_36e"
CONFIG="/workspace/project/configs/co_dino_chiayi_v14_own_dynamic_36e.py"
CONFIG_NAME="co_dino_chiayi_v14_own_dynamic_36e"
DATA_ROOT="/workspace/datasets/coco/chiayi_v14_own_dynamic_litter"
BASELINE_DIR="/workspace/hdd2_storage/chiayi_v11_36e_65vids"
BASELINE_ERROR_DIR="${BASELINE_DIR}/error_analysis_score05"
BASELINE_CONFIG="/workspace/project/configs/co_dino_chiayi_v11_36e_65vids.py"
BASELINE_DATA_ROOT="/workspace/datasets/coco/chiayi_v11_garbage_litter_corrected"

EXP_LABEL="第13次 v14"
BASELINE_LABEL="第9次 v11"
COMPARISON_NAME="comparison_v11_9th_vs_v14_13th.md"
CONFIG_DIFF_MD="/workspace/project/experiments/13th_v14_own_dynamic/config_diff_v11_vs_v14.md"
EXP_NAME="chiayi_v14_own_dynamic_36e"   # Part 1 報告標題正確值

CHECKPOINT=$(docker exec "$CONTAINER" bash -c \
    "ls ${RESULT_DIR}/best_coco_litter_precision_epoch_*.pth 2>/dev/null | tail -1")
BASELINE_CKPT=$(docker exec "$CONTAINER" bash -c \
    "ls ${BASELINE_DIR}/best_coco_litter_precision_epoch_*.pth 2>/dev/null | tail -1")
[ -z "$CHECKPOINT" ] && { echo "ERROR: 找不到 v14 best checkpoint"; exit 1; }
echo "v14 ckpt: $CHECKPOINT"
echo "baseline(v11) ckpt: ${BASELINE_CKPT:-<none>}"

# ── Part 1: 就地改標題（chiayi_v9_36e → 正確 exp name）────────────────────────
echo "=== Part 1: 就地修正報告標題 ==="
for TAG in score05 score03; do
    docker exec "$CONTAINER" bash -c \
        "f=${RESULT_DIR}/error_analysis_${TAG}/error_analysis_report.md; \
         [ -f \"\$f\" ] && sed -i 's/chiayi_v9_36e/${EXP_NAME}/g' \"\$f\" && echo \"  patched \$f\" || echo \"  [skip] \$f\""
done

# ── 清掉舊的錯名輸出 ──────────────────────────────────────────────────────────
echo "=== 清除舊錯名輸出 ==="
docker exec "$CONTAINER" bash -c \
    "rm -f ${RESULT_DIR}/error_analysis_score05/v11_8th_vs_9th_comparison.md; \
     rm -rf ${RESULT_DIR}/litter_tp_comparison/8th ${RESULT_DIR}/litter_tp_comparison/9th; \
     echo '  done'"

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

# ── Part 4: Score 分布（GPU）─────────────────────────────────────────────────
echo "=== Part 4: Score 分布 ==="
docker exec -e CUDA_VISIBLE_DEVICES=0 "$CONTAINER" python /workspace/project/tools/post_analysis_9th.py \
    --mode score-dist --config "$CONFIG" --checkpoint "$CHECKPOINT" --data-root "$DATA_ROOT" \
    --output-dir "${RESULT_DIR}/error_analysis_score05" \
    --exp-label "$EXP_LABEL"

# ── Part 5: v11(9th) vs v14(13th) 對比 ───────────────────────────────────────
echo "=== Part 5: 對比 ==="
docker exec "$CONTAINER" python /workspace/project/tools/post_analysis_9th.py --mode comparison \
    --v7-log "${BASELINE_DIR}/training_log.md" --v8-log "${RESULT_DIR}/training_log.md" \
    --v7-error-dir "$BASELINE_ERROR_DIR" --v8-error-dir "${RESULT_DIR}/error_analysis_score05" \
    --output-dir "${RESULT_DIR}/error_analysis_score05" \
    --baseline-label "$BASELINE_LABEL" --exp-label "$EXP_LABEL" \
    --comparison-name "$COMPARISON_NAME" --config-diff-md "$CONFIG_DIFF_MD"

# ── Part 6: TP 視覺化比較（GPU）──────────────────────────────────────────────
if [ -n "$BASELINE_CKPT" ]; then
    echo "=== Part 6: TP 視覺化比較 ==="
    docker exec -e CUDA_VISIBLE_DEVICES=0 "$CONTAINER" \
        python /workspace/project/tools/visualize_litter_tp_comparison_9th.py \
        --v7-checkpoint "$BASELINE_CKPT" --v8-checkpoint "$CHECKPOINT" \
        --v7-ea-dir "$BASELINE_ERROR_DIR" --v8-ea-dir "${RESULT_DIR}/error_analysis_score05" \
        --v7-config "$BASELINE_CONFIG" --v7-data-root "$BASELINE_DATA_ROOT" --v7-label "9th_v11" \
        --v8-config "$CONFIG" --v8-data-root "$DATA_ROOT" --v8-label "13th_v14" \
        --output-dir "${RESULT_DIR}/litter_tp_comparison" \
        || echo "WARNING: Part 6 失敗，跳過"
else
    echo "WARNING: 無 baseline checkpoint，跳過 Part 6。"
fi

echo ""; echo "=== Regen 全部完成 ==="
