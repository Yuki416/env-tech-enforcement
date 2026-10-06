"""畫第 12 次訓練的 val AP vs epoch（mAP@[.5:.95] 與 mAP50），標出 best epoch 與 LR drop。"""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "/path/to/data/12th_expt_v13_bgs_4ch_stem_unfreeze_36e/train.log"
OUT = "/path/to/data/12th_expt_v13_bgs_4ch_stem_unfreeze_36e/ap_vs_epoch.png"

pat = re.compile(r"Epoch\(val\) \[(\d+)\]\[\d+/\d+\].*?bbox_mAP: ([\d.]+).*?bbox_mAP_50: ([\d.]+)")
eps, mAP, mAP50 = [], [], []
for line in Path(LOG).read_text(errors="ignore").splitlines():
    m = pat.search(line)
    if m:
        eps.append(int(m.group(1))); mAP.append(float(m.group(2))); mAP50.append(float(m.group(3)))

best_i = max(range(len(mAP)), key=lambda i: mAP[i])
best_ep, best_v = eps[best_i], mAP[best_i]

plt.figure(figsize=(11, 6))
plt.plot(eps, mAP50, "-o", color="#1f77b4", label="val mAP@0.50", markersize=4)
plt.plot(eps, mAP,   "-o", color="#d62728", label="val mAP@[.5:.95]", markersize=4)

# LR drop 標線
for d in (24, 33):
    plt.axvline(d, color="gray", ls=":", lw=1)
    plt.text(d, plt.ylim()[1]*0.02, f" LR drop ep{d}", color="gray", fontsize=8, rotation=90, va="bottom")

# best epoch
plt.axvline(best_ep, color="green", ls="--", lw=1.2)
plt.scatter([best_ep], [best_v], color="green", zorder=5)
plt.annotate(f"best ep{best_ep}\nmAP={best_v:.3f}\nmAP50={mAP50[best_i]:.3f}",
             (best_ep, best_v), textcoords="offset points", xytext=(8, -38),
             color="green", fontsize=9)

plt.title("12th training (v13 BGS 4ch, stem unfrozen) - val AP vs epoch")
plt.xlabel("epoch"); plt.ylabel("AP")
plt.ylim(0, max(mAP50)*1.12); plt.xlim(0.5, max(eps)+0.5)
plt.grid(alpha=0.3); plt.legend(loc="lower right")
plt.tight_layout(); plt.savefig(OUT, dpi=130)
print(f"saved: {OUT}")
print(f"epochs parsed: {len(eps)} | best epoch {best_ep} (mAP={best_v:.3f}, mAP50={mAP50[best_i]:.3f})")
print(f"final ep{eps[-1]}: mAP={mAP[-1]:.3f} mAP50={mAP50[-1]:.3f}")
