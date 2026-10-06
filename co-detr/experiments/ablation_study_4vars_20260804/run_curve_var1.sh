#!/usr/bin/env bash
# 變因① statictail — 雙層評估的曲線資料（用 error_analysis_v9 於 test_static594）。
#   第 2 層：treatment 全 36 epoch → litter AP@0.5 曲線（診斷「靜止尾潛力有沒有被純動態 valid 埋掉」）。
#   第 1 層：16th 的 6 顆 ckpt（best=ep23 + ep32~36）同尺 → 同 epoch 對點。
# 前置：treatment 訓練完成（epoch_1..36.pth 全在）。16th 已訓練。
# 產出：RESULT/curve/curve_litter_ap50_static594.csv（model,epoch,litter_AP50）+ 各 epoch report。
# 用法（背景）：
#   nohup bash experiments/ablation_study_4vars_20260804/run_curve_var1.sh \
#     > /path/to/data/ablation_var1_statictail_36e/curve.log 2>&1 & ; tail -f ...
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONTAINER="co-detr-var1-curve"
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
R16=/workspace/hdd2_storage/16th_expt_v17_own_dynamic_122vids_36e
DS=/workspace/datasets/coco/chiayi_v17_own_dynamic_head10_manual_statictail_ablation
CT=/workspace/project/configs/co_dino_chiayi_v17_statictail_ablation_36e.py
C16=/workspace/project/configs/co_dino_chiayi_v17_own_dynamic_head10_manual_36e.py
EA=/workspace/project/tools/error_analysis_v9.py
ANN="test=annotations/test_static594.json"
mkdir_p(){ docker exec "$CONTAINER" mkdir -p "$1"; }

curve_one(){ # $1 config  $2 ckpt  $3 out_dir
  [ -z "$2" ] && return
  docker exec "$CONTAINER" bash -c "[ -f '$2' ]" || { echo "  [skip] 無 $2"; return; }
  docker exec -e CUDA_VISIBLE_DEVICES=0 "$CONTAINER" python "$EA" \
    --config "$1" --checkpoint "$2" --data-root "$DS" --output-dir "$3" \
    --splits test --score-threshold 0.5 --split-ann "$ANN" >/dev/null 2>&1 && echo "  ok $(basename $3)"
}

echo "=== 第2層：treatment epoch 1..36 於 static594 ==="
for EP in $(seq 1 36); do
  curve_one "$CT" "${RESULT}/epoch_${EP}.pth" "${RESULT}/curve/treat_ep${EP}"
done
echo "=== 第1層：16th best(23) + ep32~36 於 static594 ==="
curve_one "$C16" "${R16}/best_coco_litter_precision_epoch_23.pth" "${RESULT}/curve/16th_ep23"
for EP in 32 33 34 35 36; do
  curve_one "$C16" "${R16}/epoch_${EP}.pth" "${RESULT}/curve/16th_ep${EP}"
done

echo "=== 解析各 report 的 litter AP@0.5 → curve CSV ==="
docker exec -i "$CONTAINER" python - <<'PY'
import re, csv, glob, os
BASE="/workspace/hdd2_storage/ablation_var1_statictail_36e/curve"
rows=[]
for d in sorted(glob.glob(f"{BASE}/*_ep*")):
    rep=os.path.join(d,"error_analysis_report.md")
    if not os.path.exists(rep): continue
    model="treatment" if os.path.basename(d).startswith("treat") else "16th"
    ep=int(re.search(r"ep(\d+)", os.path.basename(d)).group(1))
    ap=None
    for line in open(rep, encoding="utf-8"):
        m=re.match(r"\|\s*litter\s*\|\s*([0-9.]+)\s*\|", line)
        if m: ap=float(m.group(1)); break
    rows.append((model, ep, ap))
rows.sort(key=lambda x:(x[0], x[1]))
out=f"{BASE}/curve_litter_ap50_static594.csv"
with open(out,"w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["model","epoch","litter_AP50"]); w.writerows(rows)
print("→", out)
for m,e,a in rows: print(f"  {m:9s} ep{e:2d}  litter_AP50={a}")
PY

docker exec "$CONTAINER" bash -c "chmod -R a+r ${RESULT}/curve 2>/dev/null || true"
echo "CURVE_VAR1_DONE"
