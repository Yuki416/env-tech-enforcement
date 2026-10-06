#!/usr/bin/env bash
# Step 2 錨點驗證：16th(RGB) 在 static594 上，用 tools/test.py dump → metrics_from_preds。
# 目的：確認「metrics_from_preds + --score-prefilter 0.5」能重現 error_analysis_v9 的 16th 數字
#       （litter 0.041 / garbage 0.049）→ 證明兩支腳本同刻度，18th(4ch) 可坐上①/④ 同尺。
# 同時印「不加預砍（全清單標準 AP）」版，讓兩種刻度差異一目了然。
# 用法（背景）：nohup bash experiments/ablation_study_4vars_20260804/run_reeval_step2_anchor.sh \
#   > /path/to/data/ablation_var3_var4_reeval/step2_anchor.log 2>&1 & ; tail -f 該 log
set -euo pipefail
PROJ=/path/to/repo/co-detr
OUTHOST=/path/to/data/ablation_var3_var4_reeval
mkdir -p "$OUTHOST"

docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v "$PROJ":/workspace/project \
  -v /path/to/data/coco:/workspace/datasets/coco \
  -v /path/to/data:/workspace/hdd2_storage \
  -e PYTHONPATH=/opt/mmdetection:/workspace/project \
  -e PYTHONUTF8=1 -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 -e CUDA_VISIBLE_DEVICES=0 \
  co-detr:latest bash -c '
set -e
V18=/workspace/datasets/coco/chiayi_v18_own_dynamic_bgs_122vids
OUT=/workspace/hdd2_storage/ablation_var3_var4_reeval
MFP=/workspace/project/experiments/ablation_study_4vars_20260804/metrics_from_preds.py
C16=/workspace/project/configs/co_dino_chiayi_v17_own_dynamic_head10_manual_36e.py
K16=/workspace/hdd2_storage/16th_expt_v17_own_dynamic_122vids_36e/best_coco_litter_precision_epoch_23.pth
mkdir -p "$OUT"

echo "=== dump 16th(RGB) 預測 @ static594（走 config test_pipeline）==="
python /opt/mmdetection/tools/test.py "$C16" "$K16" \
  --work-dir "$OUT/work_16th_static594" \
  --out "$OUT/16th_static594.pkl" \
  --cfg-options custom_hooks=[] \
    test_dataloader.dataset.data_root=$V18/ \
    test_dataloader.dataset.ann_file=annotations/test_static594.json \
    test_dataloader.dataset.data_prefix.img=test/ \
    test_evaluator.ann_file=$V18/annotations/test_static594.json

echo
echo "########## A) metrics_from_preds  ＋預砍0.5（複製 error_analysis_v9 截斷 AP）##########"
python "$MFP" --preds "$OUT/16th_static594.pkl" \
  --ann "$V18/annotations/test_static594.json" \
  --tag "16th static594 [prefilter0.5]" --score-prefilter 0.5 \
  --json-out "$OUT/16th_static594_prefilter05.json"

echo
echo "########## B) metrics_from_preds  無預砍（全清單標準 COCO AP，對照用）##########"
python "$MFP" --preds "$OUT/16th_static594.pkl" \
  --ann "$V18/annotations/test_static594.json" \
  --tag "16th static594 [full-list]" \
  --json-out "$OUT/16th_static594_fulllist.json"

echo
echo "=== 對照基準（error_analysis_v9 落盤，變因①產）==="
echo "   16th static594  litter AP@0.5=0.041  garbage AP@0.5=0.049"
echo "STEP2_ANCHOR_DONE"
'
