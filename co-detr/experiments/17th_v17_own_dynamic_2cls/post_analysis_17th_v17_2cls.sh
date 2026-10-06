#!/usr/bin/env bash
# 第17次實驗（v17 動態子集，2 類 litter+garbage）訓練後分析 —— 【只跑核心 Part 1：誤差分析】。
# （Part 2-6：AP曲線/日夜分組/score分布/vs第13次對比/TP視覺化 —— 稍後另議，見 README checklist。）
#
# 用 fork 的 2 類版 error_analysis_17th_2cls.py（共用 v9 版硬編 5 類，2 類 pred 會錯位）。
# 產物 = litter/garbage 逐框 TP/FN/FP + per-class AP@0.5 / AP@[.5:.95]（= 主判定 0.254 的同口徑）。
#   - test 跑兩份：test_static594（★主判定，同第13次尺）+ 完整 655（參考，⚠️含同片洩漏）。
#   - score 0.5 / 0.3 各一份（0.3 供 Video Inference 的 IoU 校正用，比照歷次）。
#
# 註：error_analysis 用 init_detector（非 tools/test.py）→ 不會截斷 training_log.md，無需 custom_hooks=[]。
#
# 使用方式（背景）：
#   nohup bash experiments/17th_v17_own_dynamic_2cls/post_analysis_17th_v17_2cls.sh \
#       > /path/to/data/17th_expt_v17_own_dynamic_2cls_36e/post_analysis.log 2>&1 &
#   tail -f /path/to/data/17th_expt_v17_own_dynamic_2cls_36e/post_analysis.log

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # = co-detr 根目錄
EXP=experiments/17th_v17_own_dynamic_2cls

CONTAINER="co-detr-v17-2cls-post"
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
RESULT_DIR="/workspace/hdd2_storage/17th_expt_v17_own_dynamic_2cls_36e"
CONFIG="/workspace/project/configs/co_dino_chiayi_v17_own_dynamic_2cls_litter_garbage_36e.py"
DATA_ROOT="/workspace/datasets/coco/chiayi_v17_own_dynamic_litter_head10_manual"
EA="/workspace/project/${EXP}/error_analysis_17th_2cls.py"

# ── 自動偵測 best checkpoint（save_best=coco/litter_precision → 本次 ep24）───────
CHECKPOINT=$(docker exec "$CONTAINER" bash -c \
    "ls ${RESULT_DIR}/best_coco_litter_precision_epoch_*.pth 2>/dev/null | tail -1")
if [ -z "$CHECKPOINT" ]; then
    echo "ERROR: 找不到 ${RESULT_DIR}/best_coco_litter_precision_epoch_*.pth（訓練是否完成？）"; exit 1
fi
echo "使用 v17 2類 best checkpoint: ${CHECKPOINT}"

# ── Part 1a/1b: 誤差分析（score 0.5 / 0.3，train/valid/test；test=static594 主判定）────
for THR in 0.5 0.3; do
    TAG=$(echo "$THR" | tr -d '.'); TAG="score${TAG}"
    echo "=== Part 1: Error analysis (score_thr=${THR}) → ${TAG}  [test = 594 靜止子集] ==="
    docker exec -e CUDA_VISIBLE_DEVICES=0 "$CONTAINER" python "$EA" \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" --data-root "$DATA_ROOT" \
        --output-dir "${RESULT_DIR}/error_analysis_${TAG}" \
        --splits train valid test --score-threshold "$THR" \
        --split-ann test=annotations/test_static594.json
done

# ── Part 1c: 完整 655 test（score 0.5 / 0.3；⚠️含同片洩漏，僅參考）──────────────
for THR in 0.5 0.3; do
    TAG=$(echo "$THR" | tr -d '.'); TAG="score${TAG}"
    echo "=== Part 1c: Error analysis — 完整 655 test (score_thr=${THR}) ==="
    docker exec -e CUDA_VISIBLE_DEVICES=0 "$CONTAINER" python "$EA" \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" --data-root "$DATA_ROOT" \
        --output-dir "${RESULT_DIR}/error_analysis_full655_${TAG}" \
        --splits test --score-threshold "$THR" \
        || echo "WARNING: Part 1c (${TAG}) 失敗"
done

# ── 收尾：chmod（容器 root 寫出的檔預設 0600，主機帳號才讀得到→能後續上傳/瀏覽）─────
echo "=== 收尾：chmod（讓主機帳號可讀）==="
docker exec "$CONTAINER" bash -c \
    'for d in error_analysis_score05 error_analysis_score03 error_analysis_full655_score05 error_analysis_full655_score03; do
        p="'"${RESULT_DIR}"'/$d"; [ -d "$p" ] || continue
        chmod -R a+r "$p"; find "$p" -type d -exec chmod a+rx {} +
     done'

echo ""; echo "=== Part 1 誤差分析完成 ==="
echo "  ★主判定(594) score0.5: ${RESULT_DIR}/error_analysis_score05/"
echo "   主判定(594) score0.3: ${RESULT_DIR}/error_analysis_score03/"
echo "   參考(655,含洩漏) 0.5:  ${RESULT_DIR}/error_analysis_full655_score05/"
echo "   參考(655,含洩漏) 0.3:  ${RESULT_DIR}/error_analysis_full655_score03/"
echo "POST_ANALYSIS_PART1_DONE"
