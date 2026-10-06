#!/bin/bash
# ★★ 第 15 次的「主判定」★★
# 第 15 次(v16, 99 clips) vs 第 13 次(v14, 44 clips) 在【同一份 594 靜止 test】上的 COCO 評估。
#
# 為什麼是 594 而不是 v16 完整的 655：
#   v16 test 新增的那 1 部（20240710172351_13m21s, 61 張）與 train 的 06m27s 同源（同車同車位隔 7 分鐘），
#   有已知同片洩漏，且**只存在於第 15 次**（第 13 次沒有）→ 655 不可用來宣稱進步。
#   594 = v16 test ∩ v11 原 test，兩次逐張相同、偏差互相抵消 → 這才是乾淨的對照。
#   （v16 的 annotations/test_static594.json 由 build_v16_dynamic_litter.py 產生並 assert 過 594 張/1823 框。）
#
# 對照組數字（第 13 次，best ep15）：litter AP@0.50 = 0.254 / AP@[.5:.95] = 0.087 / mAP_s = 0.322
#
# ⚠️ 兩邊 config 的 test_dataloader 都 = val_dataloader（指向 valid.json）→ 必須用 --cfg-options 導向。
# ⚠️ 一律加 custom_hooks=[]，否則 TrainingMarkdownLogger 會覆寫清空 training_log.md。
#
# 用法：bash experiments/15th_v16_own_dynamic_121vids/eval_15th_vs_13th_on_static594.sh

set -uo pipefail

RESULT_HOST=/path/to/data/15th_expt_v16_own_dynamic_121vids_36e
OUT="$RESULT_HOST/eval_static594_vs_13th"
mkdir -p "$OUT"

V16ROOT=/workspace/datasets/coco/chiayi_v16_own_dynamic_litter/
V14ROOT=/workspace/datasets/coco/chiayi_v14_own_dynamic_litter/

# 第 15 次 best ckpt（自動偵測；save_best=coco/litter_precision）
CKPT_15=$(ls "$RESULT_HOST"/best_coco_litter_precision_epoch_*.pth 2>/dev/null | tail -1)
if [ -z "$CKPT_15" ]; then
    echo "ERROR: 找不到 $RESULT_HOST/best_coco_litter_precision_epoch_*.pth（訓練完成了嗎？）"; exit 1
fi
CKPT_15_C="/workspace/hdd2_storage/15th_expt_v16_own_dynamic_121vids_36e/$(basename "$CKPT_15")"

# 第 13 次 baseline best ckpt
CKPT_13=$(ls /path/to/data/13th_expt_v14_own_dynamic_36e/best_coco_litter_precision_epoch_*.pth 2>/dev/null | tail -1)
if [ -z "$CKPT_13" ]; then
    echo "ERROR: 找不到第 13 次 baseline checkpoint"; exit 1
fi
CKPT_13_C="/workspace/hdd2_storage/13th_expt_v14_own_dynamic_36e/$(basename "$CKPT_13")"

echo "第15次 ckpt: $(basename "$CKPT_15")"
echo "第13次 ckpt: $(basename "$CKPT_13")  (baseline)"
echo "輸出:        $OUT"
echo

DOCKER="docker run --rm --gpus '\"device=0\"' --ipc=host --shm-size=16g \
  -v /path/to/repo/co-detr:/workspace/project \
  -v /path/to/data/coco:/workspace/datasets/coco \
  -v /path/to/data:/workspace/hdd2_storage \
  -e PYTHONPATH=/opt/mmdetection:/workspace/project \
  -e LANG=C.UTF-8 -e LC_ALL=C.UTF-8 -e PYTHONUTF8=1 \
  co-detr:latest"

echo "########## [1/3] ★主判定★ 第15次 v16 — 594 靜止子集 ##########"
eval $DOCKER python /opt/mmdetection/tools/test.py \
  /workspace/project/configs/co_dino_chiayi_v16_own_dynamic_36e.py \
  "$CKPT_15_C" \
  --cfg-options custom_hooks=[] \
    test_dataloader.dataset.ann_file=annotations/test_static594.json \
    test_dataloader.dataset.data_prefix.img=test/ \
    test_evaluator.ann_file=${V16ROOT}annotations/test_static594.json \
  > "$OUT/eval_15th_v16_static594.log" 2>&1
echo "  → $OUT/eval_15th_v16_static594.log"

echo "########## [2/3] baseline 第13次 v14 — 同一份 594（其 test.json 即為此 594）##########"
eval $DOCKER python /opt/mmdetection/tools/test.py \
  /workspace/project/configs/co_dino_chiayi_v14_own_dynamic_36e.py \
  "$CKPT_13_C" \
  --cfg-options custom_hooks=[] \
    test_dataloader.dataset.ann_file=annotations/test.json \
    test_dataloader.dataset.data_prefix.img=test/ \
    test_evaluator.ann_file=${V14ROOT}annotations/test.json \
  > "$OUT/eval_13th_v14_static594_baseline.log" 2>&1
echo "  → $OUT/eval_13th_v14_static594_baseline.log"

echo "########## [3/3] 參考 第15次 v16 — 完整 655 test（⚠️ 含已知同片洩漏）##########"
eval $DOCKER python /opt/mmdetection/tools/test.py \
  /workspace/project/configs/co_dino_chiayi_v16_own_dynamic_36e.py \
  "$CKPT_15_C" \
  --cfg-options custom_hooks=[] \
    test_dataloader.dataset.ann_file=annotations/test.json \
    test_dataloader.dataset.data_prefix.img=test/ \
    test_evaluator.ann_file=${V16ROOT}annotations/test.json \
  > "$OUT/eval_15th_v16_full655_REFERENCE_ONLY.log" 2>&1
echo "  → $OUT/eval_15th_v16_full655_REFERENCE_ONLY.log"
echo "     ⚠️ 此檔數字含洩漏、litter AP 虛高，引用時必須標註，不可用來宣稱進步"

echo
echo "=== 摘要 ==="
python3 - "$OUT" <<'PY'
import re, sys, pathlib
out = pathlib.Path(sys.argv[1])
FILES = [
    ("★主判定 15th v16 (594)",        "eval_15th_v16_static594.log"),
    ("  baseline 13th v14 (594)",     "eval_13th_v14_static594_baseline.log"),
    ("  參考 15th v16 (655,含洩漏)",  "eval_15th_v16_full655_REFERENCE_ONLY.log"),
]
rows = []
for label, fn in FILES:
    p = out / fn
    if not p.exists():
        rows.append((label, {})); continue
    txt = p.read_text(errors="ignore")
    d = {}
    # ⚠️ MMDet classwise 的 coco/{cls}_precision = 該類別 AP@[.50:.95]（**不是** AP@0.50）。
    #    第 13 次報告裡的 litter AP@0.50 = 0.254 來自 error_analysis_v9.py 自算的 per-class AP@0.5，
    #    不是 tools/test.py 的輸出。兩者不可混用，欄位名務必照實標。
    for k, pat in [("litter AP.5:.95", r"litter_precision:\s*([\d.]+)"),
                   ("garbage AP.5:.95", r"garbage_precision:\s*([\d.]+)"),
                   ("bbox_mAP", r"bbox_mAP:\s*([\d.]+)"),
                   ("bbox_mAP_50", r"bbox_mAP_50:\s*([\d.]+)"),
                   ("mAP_s", r"bbox_mAP_s:\s*([\d.]+)")]:
        m = re.findall(pat, txt)
        if m: d[k] = float(m[-1])
    rows.append((label, d))
keys = ["litter AP.5:.95", "garbage AP.5:.95", "bbox_mAP", "bbox_mAP_50", "mAP_s"]
print(f'{"":32s}' + "".join(f"{k:>17s}" for k in keys))
for label, d in rows:
    print(f"{label:32s}" + "".join(f'{d[k]:>17.4f}' if k in d else f'{"-":>17s}' for k in keys))
print()
print("對照（第13次 best ep15，594 靜止 test）：litter AP@[.5:.95]=0.087 / bbox_mAP=0.466 / mAP_s=0.322")
print("⚠️ classwise 的 *_precision 是 AP@[.50:.95]。報告常引用的『litter AP@0.50 = 0.254』是另一個口徑，")
print("   由 post_analysis 的 error_analysis_score05/ 產出（同處另有逐框 TP/FN/FP）。勿混用。")
PY
echo "EVAL_DONE"
