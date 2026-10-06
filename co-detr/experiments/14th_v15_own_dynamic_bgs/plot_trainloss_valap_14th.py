#!/usr/bin/env python3
"""第14次 v15（BGS 4ch）：Train Loss vs Val AP 雙軸圖。
左軸 = train loss（log scale）：loss_cls、loss_bbox（每 epoch 12 個 log 點取平均）
右軸 = val AP@[0.50:0.95]（線性）：litter、garbage
資料源：MMEngine scalars.json（訓練那次的 vis_data，非 test.py 產生的時間戳目錄）。

★ 與第13次 plot_trainloss_valap_13th.py 的差別（刻意）：
  該腳本把數值寫死在圖例/標註裡（"0.913→0.061 ↓15×"、ylim 0.105、BEST_EP=15…），
  照抄只改路徑會產生**假標籤**。這裡一律**從資料計算**，並且不預先寫入結論性文字
  （過擬合與否留待完整 post-analysis 後判讀）。
"""
import json, glob, re, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT = "/workspace/hdd2_storage/14th_expt_v15_own_dynamic_bgs_4ch_36e"
OUT = f"{RESULT}/train_loss_vs_val_ap.png"
MILESTONES = [28, 34]   # v15 config milestones=[28,34]

# ── 找「訓練」那次的 scalars.json（排除 test.py 產生的空時間戳目錄）──────────
cands = [p for p in glob.glob(f"{RESULT}/*/vis_data/scalars.json")]
scalars = None
for p in sorted(cands):
    txt = open(p).read()
    if "loss_cls" in txt and "coco/" in txt:
        scalars = p
if scalars is None:
    raise SystemExit(f"找不到含 loss_cls + coco/ 的 scalars.json（候選：{cands}）")
print("scalars:", scalars)

# ── best epoch 由 checkpoint 檔名取得（save_best=coco/litter_precision）───────
ck = sorted(glob.glob(f"{RESULT}/best_coco_litter_precision_epoch_*.pth"))
BEST_EP = int(re.search(r"epoch_(\d+)", os.path.basename(ck[-1])).group(1)) if ck else None

tr, va = [], []
for line in open(scalars):
    try:
        d = json.loads(line)
    except Exception:
        continue
    (va if any(k.startswith("coco/") for k in d) else tr).append(d)

by = defaultdict(lambda: defaultdict(list))
for d in tr:
    for k, v in d.items():
        if isinstance(v, (int, float)):
            by[d["epoch"]][k].append(v)
eps = sorted(by)
mean = lambda k: [sum(by[e][k]) / len(by[e][k]) for e in eps]

loss_cls, loss_bbox = mean("loss_cls"), mean("loss_bbox")
v_eps = list(range(1, len(va) + 1))
litter  = [d["coco/litter_precision"]  for d in va]
garbage = [d["coco/garbage_precision"] for d in va]

# ── 全部數值由資料算出（不寫死）─────────────────────────────────────────────
def drop(v):  # 起訖與倍率
    return v[0], v[-1], (v[0] / v[-1] if v[-1] else float("inf"))
c0, c1, cx = drop(loss_cls)
b0, b1, bx = drop(loss_bbox)
lit_lo, lit_hi = min(litter[1:]), max(litter[1:])     # 排除 ep1 冷啟動
gar_lo, gar_hi = min(garbage[1:]), max(garbage[1:])
ymax = max(max(litter), max(garbage)) * 1.15

fig, axL = plt.subplots(figsize=(11, 6.2))
c_cls, c_box = "#1f4e9c", "#4a90d9"
axL.plot(eps, loss_cls, color=c_cls, lw=2.2, marker="o", ms=3.2,
         label=rf"train $loss\_cls$  ({c0:.3f} $\to$ {c1:.3f}, $\downarrow${cx:.1f}$\times$)")
axL.plot(eps, loss_bbox, color=c_box, lw=2.2, marker="s", ms=3.2, ls="--",
         label=rf"train $loss\_bbox$  ({b0:.3f} $\to$ {b1:.3f}, $\downarrow${bx:.1f}$\times$)")
axL.set_yscale("log")
axL.set_xlabel("Epoch", fontsize=12)
axL.set_ylabel("Train loss  (log scale, per-epoch mean)", fontsize=12, color=c_cls)
axL.tick_params(axis="y", labelcolor=c_cls)
axL.set_xlim(0.5, max(eps) + 0.5)
axL.grid(alpha=0.25, which="both", ls=":")

axR = axL.twinx()
c_lit, c_gar = "#c0392b", "#e8913a"
axR.axhspan(lit_lo, lit_hi, color=c_lit, alpha=0.09, zorder=0)
axR.plot(v_eps, litter, color=c_lit, lw=2.4, marker="o", ms=4.5,
         label=f"val litter  AP@[.50:.95]  (ep2-{max(v_eps)} range {lit_lo:.3f}-{lit_hi:.3f})")
axR.plot(v_eps, garbage, color=c_gar, lw=2.0, marker="^", ms=4.2,
         label=f"val garbage  AP@[.50:.95]  (ep2-{max(v_eps)} range {gar_lo:.3f}-{gar_hi:.3f})")
axR.set_ylabel("Validation AP@[0.50:0.95]  (COCO per-class AP)", fontsize=12, color=c_lit)
axR.tick_params(axis="y", labelcolor=c_lit)
axR.set_ylim(0, ymax)

# ── 事實性標註（不含結論）──────────────────────────────────────────────────
if BEST_EP:
    axR.axvline(BEST_EP, color="#333", ls="--", lw=1.5, alpha=0.8)
    axR.annotate(f"selected ckpt  ep{BEST_EP}\n(save_best = litter AP)\nval litter AP={litter[BEST_EP-1]:.3f}",
                 xy=(BEST_EP, litter[BEST_EP - 1]), xytext=(BEST_EP + 5, ymax * 0.90),
                 fontsize=9, color="#333",
                 arrowprops=dict(arrowstyle="->", color="#333", lw=1.1))
for m in MILESTONES:
    axL.axvline(m, color="gray", ls=":", lw=1.2, alpha=0.7)
    axL.text(m, axL.get_ylim()[1] * 0.72, f"LR x0.1\n(ep{m})", fontsize=8,
             color="gray", ha="center", va="top")

h1, l1 = axL.get_legend_handles_labels()
h2, l2 = axR.get_legend_handles_labels()
axL.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.13),
           ncol=2, fontsize=9.4, framealpha=0.95)

axL.set_title(
    "14th run (v15 = v14 dynamic subset + BGS changemap 4ch)\n"
    "Train loss (left, log) vs Validation AP (right) per epoch",
    fontsize=12.5, pad=13)

fig.text(0.5, -0.145,
         "Val AP = COCO per-class AP over IoU 0.50:0.95 (area=all, maxDets=100), "
         "CocoMetric(classwise=True) on the v15 dynamic valid set (488 frames).\n"
         "Note: MMDetection logs this key as \"coco/<cls>_precision\", but it is AP, not precision. "
         "Train loss = mean of the per-epoch log points. Values in the legend are computed from the log.",
         ha="center", fontsize=8.2, color="#555")

plt.tight_layout()
plt.savefig(OUT, dpi=160, bbox_inches="tight", facecolor="white")
print("saved:", OUT)
print(f"loss_cls  {c0:.3f} -> {c1:.3f}  (x{cx:.1f} down)")
print(f"loss_bbox {b0:.3f} -> {b1:.3f}  (x{bx:.1f} down)")
print(f"val litter  ep2-{max(v_eps)} range {lit_lo:.3f}-{lit_hi:.3f} | best ep{BEST_EP} = {litter[BEST_EP-1]:.3f}")
print(f"val garbage ep2-{max(v_eps)} range {gar_lo:.3f}-{gar_hi:.3f}")
