#!/usr/bin/env bash
# 第18次 post-analysis 任務 1–3：雙軌評估（dynamic166 主 / static594 副），18th 4ch + 16th RGB 同一條管線。
# 做法：tools/test.py（走 config test_pipeline，4ch config 會載 changemap）--out dump 預測 → 離線 metrics_from_preds.py。
# 兩模型都在 v18 test 影像上評（16th RGB config 忽略 changemap）→ 同影像同 GT、單一方法學。
# 用法（背景）：nohup bash .../run_dual_track_eval.sh > .../dual_track_eval.log 2>&1 & ; tail -f
set -euo pipefail
PROJ=/path/to/repo/co-detr
OUTHOST=/path/to/data/18th_expt_v18_own_dynamic_bgs_4ch_36e/dual_track_eval
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
OUT=/workspace/hdd2_storage/18th_expt_v18_own_dynamic_bgs_4ch_36e/dual_track_eval
C18=/workspace/project/configs/co_dino_chiayi_v18_own_dynamic_bgs_4ch_36e.py
K18=/workspace/hdd2_storage/18th_expt_v18_own_dynamic_bgs_4ch_36e/best_coco_litter_precision_epoch_21.pth
C16=/workspace/project/configs/co_dino_chiayi_v17_own_dynamic_head10_manual_36e.py
K16=/workspace/hdd2_storage/16th_expt_v17_own_dynamic_122vids_36e/best_coco_litter_precision_epoch_23.pth

echo "=== TASK 1: eval 管線 4ch changemap 讀取 sanity（18th config, dynamic166）==="
python - <<PY
from mmengine.config import Config
from mmengine.registry import init_default_scope, MODELS
from mmengine.utils import import_modules_from_strings
from mmengine.runner import Runner
cfg=Config.fromfile("$C18"); init_default_scope("mmdet"); import_modules_from_strings(**cfg["custom_imports"])
cfg.test_dataloader["dataset"]["data_root"]="$V18/"
cfg.test_dataloader["dataset"]["ann_file"]="annotations/test_dynamic166.json"
cfg.test_dataloader["dataset"]["data_prefix"]=dict(img="test/")
cfg.test_dataloader["batch_size"]=1; cfg.test_dataloader["num_workers"]=0; cfg.test_dataloader["persistent_workers"]=False
dl=Runner.build_dataloader(cfg.test_dataloader)
dp=MODELS.build(cfg.model["data_preprocessor"])
b=next(iter(dl)); d=dp(b, training=False)
C=d["inputs"].shape[1]; frac=(d["inputs"][:,3].abs()>1e-6).float().mean().item()
print(f"[TASK1] eval 4ch: C={C} (應4)  第4通道非零={frac*100:.2f}%  ->", "OK" if C==4 and frac>0 else "FAIL")
assert C==4 and frac>0, "4ch/ changemap 讀取失敗，停止"
PY

run(){ # $1 config $2 ckpt $3 ann $4 name
  echo "=== EVAL $4  (ann=$3) ==="
  python /opt/mmdetection/tools/test.py "$1" "$2" \
    --work-dir "$OUT/work_$4" \
    --out "$OUT/$4.pkl" \
    --cfg-options custom_hooks=[] \
      test_dataloader.dataset.data_root=$V18/ \
      test_dataloader.dataset.ann_file=annotations/$3 \
      test_dataloader.dataset.data_prefix.img=test/ \
      test_evaluator.ann_file=$V18/annotations/$3
}
run "$C18" "$K18" test_dynamic166.json 18th_dynamic166
run "$C18" "$K18" test_static594.json  18th_static594
run "$C16" "$K16" test_dynamic166.json 16th_dynamic166
run "$C16" "$K16" test_static594.json  16th_static594

echo "DUAL_TRACK_EVAL_DONE  (preds pkl 在 $OUT)"
'
