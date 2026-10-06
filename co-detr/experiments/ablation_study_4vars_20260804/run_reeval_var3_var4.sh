#!/usr/bin/env bash
# Step 3+4：③④ 在統一尺（static594 + dyn7v + dyn8v）、統一截斷 AP 刻度上重評。
#   Step 3 變因③處理組：18th(4ch) → tools/test.py dump（走 v18 影像+changemap）→ metrics_from_preds --score-prefilter 0.5。
#   Step 4 變因④處理組：17th(2cls) → error_analysis_17th_2cls.py（v9 fork，本就截斷 AP）三尺。
# 16th（兩變因共用 baseline）已由變因① 用 error_analysis_v9 在三尺跑過（16th_static594/dyn7v/dyn8v_score05），本腳本不重跑。
# 錨點（Step2）已證 mfp+預砍0.5 ≡ error_analysis_v9 → 三者同刻度。
# 用法（背景）：nohup bash experiments/ablation_study_4vars_20260804/run_reeval_var3_var4.sh \
#   > /path/to/data/ablation_var3_var4_reeval/var3_var4.log 2>&1 & ; tail -f 該 log
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
DS1=/workspace/datasets/coco/chiayi_v17_own_dynamic_head10_manual_statictail_ablation
OUT=/workspace/hdd2_storage/ablation_var3_var4_reeval
MFP=/workspace/project/experiments/ablation_study_4vars_20260804/metrics_from_preds.py
C18=/workspace/project/configs/co_dino_chiayi_v18_own_dynamic_bgs_4ch_36e.py
K18=/workspace/hdd2_storage/18th_expt_v18_own_dynamic_bgs_4ch_36e/best_coco_litter_precision_epoch_21.pth
C17=/workspace/project/configs/co_dino_chiayi_v17_own_dynamic_2cls_litter_garbage_36e.py
K17=/workspace/hdd2_storage/17th_expt_v17_own_dynamic_2cls_36e/best_coco_litter_precision_epoch_24.pth
EA17=/workspace/project/experiments/17th_v17_own_dynamic_2cls/error_analysis_17th_2cls.py
mkdir -p "$OUT"

########################## Step 3：18th(4ch) 三尺 ##########################
dump18(){ # $1 ann_basename  $2 name
  echo "=== [18th] dump $2  (ann=$1) ==="
  python /opt/mmdetection/tools/test.py "$C18" "$K18" \
    --work-dir "$OUT/work_$2" --out "$OUT/$2.pkl" \
    --cfg-options custom_hooks=[] \
      test_dataloader.dataset.data_root=$V18/ \
      test_dataloader.dataset.ann_file=annotations/$1 \
      test_dataloader.dataset.data_prefix.img=test/ \
      test_evaluator.ann_file=$V18/annotations/$1
  echo "--- [18th] metrics_from_preds +預砍0.5  $2 ---"
  python "$MFP" --preds "$OUT/$2.pkl" --ann "$V18/annotations/$1" \
    --tag "18th $2 [prefilter0.5]" --score-prefilter 0.5 \
    --json-out "$OUT/$2_metrics.json"
}
dump18 test_static594.json         18th_static594
dump18 test_dynamic_v17_7vids.json 18th_dyn7v
dump18 test_dynamic_v17_8vids.json 18th_dyn8v

########################## Step 4：17th(2cls) 三尺 ##########################
ea17(){ # $1 ann_basename  $2 outdir_name
  echo "=== [17th 2cls] error_analysis  $2  (ann=$1) ==="
  python "$EA17" \
    --config "$C17" --checkpoint "$K17" --data-root "$DS1" \
    --output-dir "$OUT/$2" --splits test --score-threshold 0.5 \
    --split-ann test=annotations/$1
}
ea17 test_static594.json         17th_static594
ea17 test_dynamic_v17_7vids.json 17th_dyn7v
ea17 test_dynamic_v17_8vids.json 17th_dyn8v

chmod -R a+r "$OUT" 2>/dev/null || true
echo "REEVAL_VAR3_VAR4_DONE  (18th pkl+metrics.json、17th error_analysis 在 $OUT)"
'
