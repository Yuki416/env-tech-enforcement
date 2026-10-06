#!/bin/bash
# 第 14 次(v15 BGS 4ch) vs 第 13 次(v14 純RGB) 在「同一份 594 靜止 test」上的 COCO 評估。
# 兩邊各用自己的 config（v15 test 影像/標註 = v14 原封複製，逐張相同）。
# ⚠️ 第13次 config 的 test_dataloader = val_dataloader（指向 valid.json）→ 必須用 --cfg-options 導到 test.json。
# ⚠️ 一律加 custom_hooks=[]，否則 TrainingMarkdownLogger 會覆寫清空 training_log.md。

OUT=/path/to/data/14th_expt_v15_own_dynamic_bgs_4ch_36e/eval_static_test_vs_13th
mkdir -p "$OUT"

DOCKER="docker run --rm --gpus '\"device=0\"' --ipc=host --shm-size=16g \
  -v /path/to/repo/co-detr:/workspace/project \
  -v /path/to/data/coco:/workspace/datasets/coco \
  -v /path/to/data:/workspace/hdd2_storage \
  -e PYTHONPATH=/opt/mmdetection:/workspace/project \
  -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 -e PYTHONUTF8=1 \
  co-detr:latest"

echo "########## [1/2] 第14次 v15 BGS 4ch — best epoch 12 ##########"
eval $DOCKER python /opt/mmdetection/tools/test.py \
  /workspace/project/configs/co_dino_chiayi_v15_own_dynamic_bgs_4ch_36e.py \
  /workspace/hdd2_storage/14th_expt_v15_own_dynamic_bgs_4ch_36e/best_coco_litter_precision_epoch_12.pth \
  --cfg-options custom_hooks=[] \
  > "$OUT/eval_14th_v15.log" 2>&1
echo "  → $OUT/eval_14th_v15.log"

echo "########## [2/2] 第13次 v14 純RGB — best epoch 15（baseline，導向 test.json）##########"
V14ROOT=/workspace/datasets/coco/chiayi_v14_own_dynamic_litter/
eval $DOCKER python /opt/mmdetection/tools/test.py \
  /workspace/project/configs/co_dino_chiayi_v14_own_dynamic_36e.py \
  /workspace/hdd2_storage/13th_expt_v14_own_dynamic_36e/best_coco_litter_precision_epoch_15.pth \
  --cfg-options custom_hooks=[] \
    test_dataloader.dataset.ann_file=annotations/test.json \
    test_dataloader.dataset.data_prefix.img=test/ \
    test_evaluator.ann_file=${V14ROOT}annotations/test.json \
  > "$OUT/eval_13th_v14_baseline.log" 2>&1
echo "  → $OUT/eval_13th_v14_baseline.log"

echo "ALL_EVAL_DONE"
