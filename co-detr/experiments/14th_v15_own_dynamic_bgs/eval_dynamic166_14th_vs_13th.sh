#!/bin/bash
# 輔助判定：第14次(v15 BGS 4ch) vs 第13次(v14 純RGB) 在「166 張動態子集」上的 COCO 評估。
# 主判定是 594 靜止 test（eval_14th_vs_13th_on_static_test.sh）。這裡問的是：
#   「BGS 在它**有訊號**的地方（垃圾仍在動）表現如何？」→ 用來區辨
#   「BGS 這條路無效」 vs 「BGS 只在有 motion 時有效，但我們的任務主體是靜止垃圾」。
# 標註 = test_dynamic166.json（由 build_dynamic_subset_ann.py 從 594 test 過濾出的同一批動態幀，5類GT）。
# ⚠️ v15 側 data_root 必須是 v15（有 changemap）；v14 側用 v14。兩者影像逐張相同。
# ⚠️ 一律加 custom_hooks=[]，否則 TrainingMarkdownLogger 會覆寫清空 training_log.md。

set -euo pipefail
OUT=/path/to/data/14th_expt_v15_own_dynamic_bgs_4ch_36e/eval_dynamic166_vs_13th
mkdir -p "$OUT"

DOCKER_RUN() {
  docker run --rm --gpus '"device=0"' --ipc=host --shm-size=16g \
    -v /path/to/repo/co-detr:/workspace/project \
    -v /path/to/data/coco:/workspace/datasets/coco \
    -v /path/to/data:/workspace/hdd2_storage \
    -e PYTHONPATH=/opt/mmdetection:/workspace/project \
    -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 -e PYTHONUTF8=1 \
    co-detr:latest "$@"
}

V15ROOT=/workspace/datasets/coco/chiayi_v15_own_dynamic_bgs/
V14ROOT=/workspace/datasets/coco/chiayi_v14_own_dynamic_litter/

echo "########## [1/2] 第14次 v15 BGS 4ch — best epoch 12（166 動態子集）##########"
DOCKER_RUN python /opt/mmdetection/tools/test.py \
  /workspace/project/configs/co_dino_chiayi_v15_own_dynamic_bgs_4ch_36e.py \
  /workspace/hdd2_storage/14th_expt_v15_own_dynamic_bgs_4ch_36e/best_coco_litter_precision_epoch_12.pth \
  --cfg-options custom_hooks=[] \
    test_dataloader.dataset.ann_file=annotations/test_dynamic166.json \
    test_evaluator.ann_file=${V15ROOT}annotations/test_dynamic166.json \
  > "$OUT/eval_14th_v15_dynamic166.log" 2>&1
echo "  → $OUT/eval_14th_v15_dynamic166.log"

echo "########## [2/2] 第13次 v14 純RGB — best epoch 15（166 動態子集，baseline）##########"
DOCKER_RUN python /opt/mmdetection/tools/test.py \
  /workspace/project/configs/co_dino_chiayi_v14_own_dynamic_36e.py \
  /workspace/hdd2_storage/13th_expt_v14_own_dynamic_36e/best_coco_litter_precision_epoch_15.pth \
  --cfg-options custom_hooks=[] \
    test_dataloader.dataset.ann_file=annotations/test_dynamic166.json \
    test_dataloader.dataset.data_prefix.img=test/ \
    test_evaluator.ann_file=${V14ROOT}annotations/test_dynamic166.json \
  > "$OUT/eval_13th_v14_dynamic166.log" 2>&1
echo "  → $OUT/eval_13th_v14_dynamic166.log"

echo "DYNAMIC166_EVAL_DONE"
