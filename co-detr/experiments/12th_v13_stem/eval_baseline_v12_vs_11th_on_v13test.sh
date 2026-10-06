#!/usr/bin/env bash
# 在「同一份 v13 動態 test」上做公平 COCO 評估，回答「BGS 4ch 到底有沒有用」：
#   A) v12 best epoch 17（純 RGB 3ch）  ← baseline
#   B) 11th best epoch 30（RGB+changemap 4ch）
# 兩者皆 test.py + custom_hooks=[]（避免 TrainingMarkdownLogger 清空 training_log.md）。
set -euo pipefail

PROJECT_DIR="/path/to/repo/co-detr"
HDD="/path/to/data"
OUT="/path/to/data/baseline_v12_vs_11th_on_v13_dynamic_test"
V13="/workspace/datasets/coco/chiayi_v13_other_team_litter_dynamic_test_frame"
mkdir -p "${OUT}/v12_rgb_eval" "${OUT}/11th_4ch_eval"

run() { docker run --rm --gpus '"device=0"' --ipc=host --shm-size=16g \
  -v "${PROJECT_DIR}:/workspace/project" \
  -v "${HDD}/user/coco:/workspace/datasets/coco" \
  -v "${HDD}/user:/workspace/hdd2_storage" \
  -e PYTHONPATH=/opt/mmdetection:/workspace/project \
  -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 \
  co-detr:latest "$@"; }

echo "==================== [A] v12 RGB baseline on v13 dynamic test ===================="
run python /opt/mmdetection/tools/test.py \
  /workspace/project/configs/co_dino_chiayi_v12_litter_only_36e.py \
  /workspace/hdd2_storage/10th_expt_v12_other_team_litter_36e/best_coco_bbox_mAP_epoch_17.pth \
  --work-dir /workspace/hdd2_storage/baseline_v12_vs_11th_on_v13_dynamic_test/v12_rgb_eval \
  --cfg-options \
    test_dataloader.dataset.data_root="${V13}/" \
    test_dataloader.dataset.ann_file=annotations/test.json \
    test_dataloader.dataset.data_prefix.img=test/ \
    test_evaluator.ann_file="${V13}/annotations/test.json" \
    custom_hooks="[]"

echo "==================== [B] 11th 4ch on v13 dynamic test ===================="
run python /opt/mmdetection/tools/test.py \
  /workspace/project/configs/co_dino_chiayi_v13_4ch_litter_36e.py \
  /workspace/hdd2_storage/11th_expt_v13_bgs_4ch_litter_36e/best_coco_bbox_mAP_epoch_30.pth \
  --work-dir /workspace/hdd2_storage/baseline_v12_vs_11th_on_v13_dynamic_test/11th_4ch_eval \
  --cfg-options custom_hooks="[]"

echo "==================== DONE ===================="
