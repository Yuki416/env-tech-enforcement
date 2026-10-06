"""畫變因① statictail 訓練的 val litter / garbage AP vs epoch，標各自 best epoch 與 LR drop。
沿用第 16 次 plot_ap_vs_epoch_16th.py 的畫法（每次實驗各自獨立腳本、不共用硬編路徑）。

口徑：讀 train.log 的 coco/litter_precision、coco/garbage_precision = COCO AP@[0.50:0.95]，與 16th 同尺。"""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "/path/to/data/ablation_var1_statictail_36e/train.log"
OUT = "/path/to/data/ablation_var1_statictail_36e/ap_vs_epoch.png"
LR_DROPS = (28, 34)  # milestones=[28,34]，與 16th 相同

pat = re.compile(r"Epoch\(val\) \[(\d+)\]\[\d+/\d+\].*?litter_precision: ([\d.]+).*?garbage_precision: ([\d.]+)")
eps, litter, garbage = [], [], []
for line in Path(LOG).read_text(errors="ignore").splitlines():
    m = pat.search(line)
    if m:
        eps.append(int(m.group(1))); litter.append(float(m.group(2))); garbage.append(float(m.group(3)))

if not eps:
    raise SystemExit(f"未從 log 解析到任何 Epoch(val) 行：{LOG}")

l_best_i = max(range(len(litter)), key=lambda i: litter[i])
g_best_i = max(range(len(garbage)), key=lambda i: garbage[i])

plt.figure(figsize=(11, 6))
plt.plot(eps, litter,  "-o", color="#d62728", label="val litter AP@[.50:.95]",  markersize=4)
plt.plot(eps, garbage, "-o", color="#1f77b4", label="val garbage AP@[.50:.95]", markersize=4)

for d in LR_DROPS:
    plt.axvline(d, color="gray", ls=":", lw=1)
    plt.text(d, plt.ylim()[1]*0.02, f" LR drop ep{d}", color="gray", fontsize=8, rotation=90, va="bottom")

for i, vals, color, name in ((l_best_i, litter, "#d62728", "litter"),
                             (g_best_i, garbage, "#1f77b4", "garbage")):
    plt.axvline(eps[i], color=color, ls="--", lw=1.0, alpha=0.6)
    plt.scatter([eps[i]], [vals[i]], color=color, zorder=5)
    plt.annotate(f"{name} best ep{eps[i]}\nAP={vals[i]:.3f}",
                 (eps[i], vals[i]), textcoords="offset points", xytext=(8, 6),
                 color=color, fontsize=9)

plt.title("Ablation var1 statictail - val litter/garbage AP@[.50:.95] vs epoch")
plt.xlabel("epoch"); plt.ylabel("AP@[0.50:0.95] (COCO mAP)")
plt.ylim(0, max(max(litter), max(garbage)) * 1.20); plt.xlim(0.5, max(eps) + 0.5)
plt.grid(alpha=0.3); plt.legend(loc="upper right")
plt.tight_layout(); plt.savefig(OUT, dpi=130)
print(f"saved: {OUT}")
print(f"epochs parsed: {len(eps)}")
print(f"litter  best ep{eps[l_best_i]} AP={litter[l_best_i]:.3f} | final ep{eps[-1]} AP={litter[-1]:.3f}")
print(f"garbage best ep{eps[g_best_i]} AP={garbage[g_best_i]:.3f} | final ep{eps[-1]} AP={garbage[-1]:.3f}")
