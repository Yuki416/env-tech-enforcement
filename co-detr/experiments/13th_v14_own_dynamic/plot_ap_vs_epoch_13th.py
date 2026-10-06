"""畫第 13 次訓練（v14 own dynamic litter）的 val 上 litter / garbage AP vs epoch，
標出各自 best epoch 與 LR drop。沿用第 12 次 plot_ap_vs_epoch_12th.py 的畫法慣例。"""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "/path/to/data/13th_expt_v14_own_dynamic_36e/train.log"
OUT = "/path/to/data/13th_expt_v14_own_dynamic_36e/ap_vs_epoch.png"
LR_DROPS = (28, 34)  # v14 milestones=[28,34]

# 每個 val epoch 取 litter / garbage 的 per-class AP（log 欄位 coco/{cls}_precision）
pat = re.compile(r"Epoch\(val\) \[(\d+)\]\[\d+/\d+\].*?litter_precision: ([\d.]+).*?garbage_precision: ([\d.]+)")
eps, litter, garbage = [], [], []
for line in Path(LOG).read_text(errors="ignore").splitlines():
    m = pat.search(line)
    if m:
        eps.append(int(m.group(1))); litter.append(float(m.group(2))); garbage.append(float(m.group(3)))

l_best_i = max(range(len(litter)), key=lambda i: litter[i])
g_best_i = max(range(len(garbage)), key=lambda i: garbage[i])

plt.figure(figsize=(11, 6))
plt.plot(eps, litter,  "-o", color="#d62728", label="val litter AP@[.50:.95]",  markersize=4)
plt.plot(eps, garbage, "-o", color="#1f77b4", label="val garbage AP@[.50:.95]", markersize=4)

# LR drop 標線
for d in LR_DROPS:
    plt.axvline(d, color="gray", ls=":", lw=1)
    plt.text(d, plt.ylim()[1]*0.02, f" LR drop ep{d}", color="gray", fontsize=8, rotation=90, va="bottom")

# 各自 best epoch
for i, vals, color, name in ((l_best_i, litter, "#d62728", "litter"),
                              (g_best_i, garbage, "#1f77b4", "garbage")):
    plt.axvline(eps[i], color=color, ls="--", lw=1.0, alpha=0.6)
    plt.scatter([eps[i]], [vals[i]], color=color, zorder=5)
    plt.annotate(f"{name} best ep{eps[i]}\nAP={vals[i]:.3f}",
                 (eps[i], vals[i]), textcoords="offset points", xytext=(8, 6),
                 color=color, fontsize=9)

plt.title("13th training (v14 own dynamic litter) - val litter/garbage AP@[.50:.95] vs epoch")
plt.xlabel("epoch"); plt.ylabel("AP@[0.50:0.95] (COCO mAP)")
plt.ylim(0, max(max(litter), max(garbage)) * 1.20); plt.xlim(0.5, max(eps) + 0.5)
plt.grid(alpha=0.3); plt.legend(loc="upper right")
plt.tight_layout(); plt.savefig(OUT, dpi=130)
print(f"saved: {OUT}")
print(f"epochs parsed: {len(eps)}")
print(f"litter  best ep{eps[l_best_i]} AP={litter[l_best_i]:.3f} | final ep{eps[-1]} AP={litter[-1]:.3f}")
print(f"garbage best ep{eps[g_best_i]} AP={garbage[g_best_i]:.3f} | final ep{eps[-1]} AP={garbage[-1]:.3f}")
