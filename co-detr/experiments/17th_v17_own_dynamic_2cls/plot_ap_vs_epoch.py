"""畫第 17 次訓練（v17 own dynamic，2 類 litter+garbage）的 val litter / garbage AP vs epoch，
標出各自 best epoch 與 LR drop。**沿用第 13/14 次 plot_ap_vs_epoch_*.py 的畫法慣例**：
  - metric = coco/{cls}_precision = **AP@[0.50:0.95]**（COCO mAP，IoU 0.5→0.95 平均；不套 score 門檻）。
  - litter=紅 #d62728 / garbage=藍 #1f77b4，-o markersize=4，LR drop 虛線 + best-epoch 標註。
（每次實驗各自獨立腳本，不共用前次硬編路徑。）"""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "/path/to/data/17th_expt_v17_own_dynamic_2cls_36e/train.log"
OUT = "/workspace/project/experiments/17th_v17_own_dynamic_2cls/ap_vs_epoch.png"
LR_DROPS = (24, 33)  # 17th milestones=[24,33]（照第 10 次配方）

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

plt.title("17th training (v17 own dynamic, 2-class litter+garbage) - val litter/garbage AP@[.50:.95] vs epoch")
plt.xlabel("epoch"); plt.ylabel("AP@[0.50:0.95] (COCO mAP)")
plt.ylim(0, max(max(litter), max(garbage)) * 1.20); plt.xlim(0.5, max(eps) + 0.5)
plt.grid(alpha=0.3); plt.legend(loc="upper right")
plt.tight_layout(); plt.savefig(OUT, dpi=130)
print(f"saved: {OUT}")
print(f"epochs parsed: {len(eps)}")
print(f"litter  best ep{eps[l_best_i]} AP={litter[l_best_i]:.3f} | final ep{eps[-1]} AP={litter[-1]:.3f}")
print(f"garbage best ep{eps[g_best_i]} AP={garbage[g_best_i]:.3f} | final ep{eps[-1]} AP={garbage[-1]:.3f}")
