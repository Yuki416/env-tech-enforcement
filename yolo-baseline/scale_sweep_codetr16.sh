#!/usr/bin/env bash
# 縮放實驗（《數據彙整（End-to-End Rerun）》第 14 節）：第 16 次 Co-DETR 在 valid 上，以不同輸入尺寸推論。
# 與 scale_sweep_yolo.py 對照。做法和 run_codetr16_valid_preds.sh 相同（同 config、同權重、tools/test.py --out），
# 只把 test pipeline 的 Resize scale 與 Pad size 換成 512 / 768 / 1024 / 1536（長邊，keep_ratio）。
# test pipeline 的順序：0 LoadImageFromFile、1 Resize、2 Pad、3 LoadAnnotations、4 PackDetInputs。
# 1024 是正式設定，重跑一次用來確認與 codetr16_valid/16th_valid.pkl 一致。
# ⚠️ 一定要 custom_hooks=[]：config 掛了 TrainingMarkdownLogger，會清空該實驗的 training_log.md。
#
# 用法：bash scale_sweep_codetr16.sh <GPU 編號>
set -euo pipefail
GPU="${1:?用法：bash scale_sweep_codetr16.sh <GPU 編號>}"
PROJ=/path/to/repo/co-detr

docker run --rm --gpus "\"device=${GPU}\"" --ipc=host --shm-size=16g \
  -v "$PROJ":/workspace/project \
  -v /path/to/data/coco:/workspace/datasets/coco \
  -v /path/to/data:/workspace/hdd2_storage \
  -e PYTHONPATH=/opt/mmdetection:/workspace/project \
  -e PYTHONUTF8=1 -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 \
  co-detr:latest bash -c '
set -e
V17=/workspace/datasets/coco/chiayi_v17_own_dynamic_litter_head10_manual
C16=/workspace/project/configs/co_dino_chiayi_v17_own_dynamic_head10_manual_36e.py
K16=/workspace/hdd2_storage/16th_expt_v17_own_dynamic_122vids_36e/best_coco_litter_precision_epoch_23.pth
for SZ in 512 768 1024 1536; do
  OUT=/workspace/hdd2_storage/baseline_yolo26l_v17/e2e/scale_sweep/16th_${SZ}
  if [ -f "$OUT/valid.pkl" ]; then echo "skip $SZ"; continue; fi
  mkdir -p "$OUT"
  python /opt/mmdetection/tools/test.py "$C16" "$K16" \
    --work-dir "$OUT/work" \
    --out "$OUT/valid.pkl" \
    --cfg-options custom_hooks=[] \
      test_dataloader.dataset.data_root=$V17/ \
      test_dataloader.dataset.ann_file=annotations/valid.json \
      test_dataloader.dataset.data_prefix.img=valid/ \
      test_evaluator.ann_file=$V17/annotations/valid.json \
      "test_dataloader.dataset.pipeline.1.scale=($SZ,$SZ)" \
      "test_dataloader.dataset.pipeline.2.size=($SZ,$SZ)"
  echo "→ $OUT/valid.pkl"
done
chown -R '"$(id -u):$(id -g)"' /workspace/hdd2_storage/baseline_yolo26l_v17/e2e/scale_sweep
echo SCALE_SWEEP_CODETR16_DONE
'
