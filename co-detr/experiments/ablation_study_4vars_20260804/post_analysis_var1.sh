#!/usr/bin/env bash
# 變因① statictail — Error Analysis（★ error analysis 第一步，沿用第16次 post_analysis Part 1）。
# 用 tools/error_analysis_v9.py 對 best ckpt 跑推論 → 產 AP@0.5/TP/FP/FN 報告 + {split}_per_frame_class_errors.csv。
#   後者是 Video Inference 第二步(fix) 的輸入 → 必須先跑完本腳本，再跑 run_video_inference_var1.sh。
# 純 RGB：treatment 與 16th 同一支 error_analysis_v9 → 同腳本可比（就像 16th Part 5 對比 baseline）。
#
# 產出目錄（RESULT=/path/to/data/ablation_var1_statictail_36e/）：
#   error_analysis_score05 / _score03      : treatment，train/valid/test(=static594 主判定)
#   error_analysis_full655_score05 / _03   : treatment，test 完整 655（video fix 的 test 對應 + 洩漏參考）
#   error_analysis_dyn8v_score05 / dyn7v   : treatment，dynamic 8v/7v
#   16th_static594_score05 / full655 / dyn8v / dyn7v : 16th best 於同尺（主判定對照）
#
# 用法（背景）：
#   nohup bash experiments/ablation_study_4vars_20260804/post_analysis_var1.sh \
#     > /path/to/data/ablation_var1_statictail_36e/post_analysis.log 2>&1 & ; tail -f ...
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # co-detr 根目錄
CONTAINER="co-detr-var1-ea"
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

RESULT=/workspace/hdd2_storage/ablation_var1_statictail_36e
DS=/workspace/datasets/coco/chiayi_v17_own_dynamic_head10_manual_statictail_ablation
CT=/workspace/project/configs/co_dino_chiayi_v17_statictail_ablation_36e.py
C16=/workspace/project/configs/co_dino_chiayi_v17_own_dynamic_head10_manual_36e.py
K16=/workspace/hdd2_storage/16th_expt_v17_own_dynamic_122vids_36e/best_coco_litter_precision_epoch_23.pth
EA=/workspace/project/tools/error_analysis_v9.py

KT=$(docker exec "$CONTAINER" bash -c "ls ${RESULT}/best_coco_litter_precision_epoch_*.pth 2>/dev/null | tail -1")
[ -z "$KT" ] && { echo "ERROR: 找不到 treatment best ckpt，訓練完成了嗎？"; exit 1; }
echo "treatment best = $KT"

ea(){ # $1 config  $2 ckpt  $3 output_dir  $4 splits(空白分隔)  $5 score  $6 split-ann(可空)
  local extra=""; [ -n "$6" ] && extra="--split-ann $6"
  echo "=== error_analysis: out=$(basename $3) splits=[$4] score=$5 $extra ==="
  docker exec -e CUDA_VISIBLE_DEVICES=0 "$CONTAINER" python "$EA" \
    --config "$1" --checkpoint "$2" --data-root "$DS" --output-dir "$3" \
    --splits $4 --score-threshold "$5" $extra
}

# ── TREATMENT：主判定 static594（train/valid/test）score 0.5 + 0.3 ──────────────
for THR in 0.5 0.3; do
  TAG="score$(echo $THR | tr -d '.')"
  ea "$CT" "$KT" "$RESULT/error_analysis_$TAG" "train valid test" "$THR" "test=annotations/test_static594.json"
done
# ── TREATMENT：完整 655 test（video fix 的 test 對應）score 0.5 + 0.3 ───────────
for THR in 0.5 0.3; do
  TAG="score$(echo $THR | tr -d '.')"
  ea "$CT" "$KT" "$RESULT/error_analysis_full655_$TAG" "test" "$THR" ""
done
# ── TREATMENT：dynamic 8v / 7v（score 0.5）──────────────────────────────────────
ea "$CT" "$KT" "$RESULT/error_analysis_dyn8v_score05" "test" "0.5" "test=annotations/test_dynamic_v17_8vids.json"
ea "$CT" "$KT" "$RESULT/error_analysis_dyn7v_score05" "test" "0.5" "test=annotations/test_dynamic_v17_7vids.json"

# ── 16th best 於同尺（主判定對照，score 0.5）────────────────────────────────────
ea "$C16" "$K16" "$RESULT/16th_static594_score05" "test" "0.5" "test=annotations/test_static594.json"
ea "$C16" "$K16" "$RESULT/16th_full655_score05"   "test" "0.5" ""
ea "$C16" "$K16" "$RESULT/16th_dyn8v_score05"      "test" "0.5" "test=annotations/test_dynamic_v17_8vids.json"
ea "$C16" "$K16" "$RESULT/16th_dyn7v_score05"      "test" "0.5" "test=annotations/test_dynamic_v17_7vids.json"

docker exec "$CONTAINER" bash -c "chmod -R a+r $RESULT/error_analysis_* $RESULT/16th_*_score05 2>/dev/null || true"
echo "POST_ANALYSIS_VAR1_DONE"
