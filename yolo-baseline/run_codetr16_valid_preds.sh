#!/usr/bin/env bash
# 步驟 6 前置：讓第 16 次 Co-DETR 在 valid 上推論一次，產出預測 pkl，用來在 valid 上找第 16 次的 best-F1 門檻。
# （第 16 次的 test 預測已經有了：第 18 次 dual_track_eval 的 16th_dynamic166.pkl / 16th_static594.pkl，不必重跑。）
#
# 做法與第 18 次的 run_dual_track_eval.sh 相同（tools/test.py --out），只是 ann 換成 valid。
# ⚠️ 一定要 custom_hooks=[]：第 16 次 config 掛了 TrainingMarkdownLogger，before_run 會以覆寫模式清空該實驗的 training_log.md。
#
# 用法：nohup bash run_codetr16_valid_preds.sh > /path/to/data/baseline_yolo26l_v17/logs/codetr16_valid_preds.log 2>&1 &
set -euo pipefail
PROJ=/path/to/repo/co-detr
OUTHOST=/path/to/data/baseline_yolo26l_v17/codetr16_valid
mkdir -p "$OUTHOST"

docker run --rm --gpus '"device=1"' --ipc=host --shm-size=16g \
  -v "$PROJ":/workspace/project \
  -v /path/to/data/coco:/workspace/datasets/coco \
  -v /path/to/data:/workspace/hdd2_storage \
  -e PYTHONPATH=/opt/mmdetection:/workspace/project \
  -e PYTHONUTF8=1 -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 \
  co-detr:latest bash -c '
set -e
V17=/workspace/datasets/coco/chiayi_v17_own_dynamic_litter_head10_manual
OUT=/workspace/hdd2_storage/baseline_yolo26l_v17/codetr16_valid
C16=/workspace/project/configs/co_dino_chiayi_v17_own_dynamic_head10_manual_36e.py
K16=/workspace/hdd2_storage/16th_expt_v17_own_dynamic_122vids_36e/best_coco_litter_precision_epoch_23.pth
python /opt/mmdetection/tools/test.py "$C16" "$K16" \
  --work-dir "$OUT/work" \
  --out "$OUT/16th_valid.pkl" \
  --cfg-options custom_hooks=[] \
    test_dataloader.dataset.data_root=$V17/ \
    test_dataloader.dataset.ann_file=annotations/valid.json \
    test_dataloader.dataset.data_prefix.img=valid/ \
    test_evaluator.ann_file=$V17/annotations/valid.json
# docker 以 root 寫出，改回一般使用者擁有，之後才能管理這些檔案
chown -R '"$(id -u):$(id -g)"' "$OUT"
echo CODETR16_VALID_DONE
'
