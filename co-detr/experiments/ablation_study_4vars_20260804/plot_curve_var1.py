#!/usr/bin/env python3
# 變因① 第2層診斷曲線：treatment 全 36 epoch 於 test_static594 的 litter AP@0.5。
# 疊上 16th（Layer1 同 epoch 對點：ep23=best + ep32~36）。
# 標註：LR drop(28,34)、treatment valid 選中的 ep15(埋掉的點)、treatment 峰值、16th best(ep23)。
# 在 docker 內跑（host matplotlib 壞）：
#   docker run --rm -v /path/to/data:/workspace/hdd2_storage co-detr:latest \
#     python /workspace/project/experiments/ablation_study_4vars_20260804/plot_curve_var1.py
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/workspace/hdd2_storage/ablation_var1_statictail_36e/curve"
CSV = f"{BASE}/curve_litter_ap50_static594.csv"

treat, sixteenth = {}, {}
with open(CSV, encoding="utf-8") as f:
    for r in csv.DictReader(f):
        ep = int(r["epoch"]); ap = float(r["litter_AP50"])
        (treat if r["model"] == "treatment" else sixteenth)[ep] = ap

te = sorted(treat); ta = [treat[e] for e in te]
peak_ep = max(treat, key=treat.get); peak_ap = treat[peak_ep]

fig, ax = plt.subplots(figsize=(11, 6.4))
ax.set_ylim(0, 0.44)          # headroom so peak label clears the title
ax.set_xlim(-0.5, 37)
# treatment full curve
ax.plot(te, ta, "-o", color="#1f77b4", lw=1.8, ms=4, label="treatment (static-tail), all 36 epochs")
# 16th same-epoch points
se = sorted(sixteenth); sa = [sixteenth[e] for e in se]
ax.plot(se, sa, "s", color="#d62728", ms=9, label="16th (pure-dynamic), same-epoch points")

# LR drops (label near bottom to avoid the legend / title zone)
for x in (28, 34):
    ax.axvline(x, color="gray", ls="--", lw=1, alpha=0.6)
    ax.text(x + 0.15, 0.40, f"LR drop ep{x}", rotation=90,
            va="top", ha="left", fontsize=8, color="gray")

# annotate key points (place text boxes in empty regions, arrow to the point)
def annot(txt, ep, ap, tx, ty, color):
    ax.annotate(txt, (ep, ap), xytext=(tx, ty), fontsize=9, color=color,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, alpha=0.9),
                arrowprops=dict(arrowstyle="->", color=color, lw=1))
annot(f"peak ep{peak_ep}={peak_ap:.3f}", peak_ep, peak_ap, 18.5, 0.38, "#1f77b4")
annot(f"valid-selected ep15={treat[15]:.3f} (buried)", 15, treat[15], 16.5, 0.135, "#7f0000")
annot(f"16th best ep23={sixteenth[23]:.3f}", 23, sixteenth[23], 24.0, 0.16, "#d62728")

ax.set_xlabel("Epoch"); ax.set_ylabel("litter AP@0.5  (test_static594)")
ax.set_title("Variable 1 (static-tail) - Layer-2 diagnostic: litter AP@0.5 on static test vs epoch\n"
             "(pure-dynamic valid selects only ep15, burying the static-tail's static-test potential)",
             pad=12)
ax.set_xticks(range(0, 37, 2))
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", framealpha=0.95)
fig.tight_layout()
out = f"{BASE}/curve_litter_ap50_static594.png"
fig.savefig(out, dpi=130)
os.chmod(out, 0o644)
print("saved", out)
print(f"treatment peak ep{peak_ep}={peak_ap:.3f} | ep15={treat[15]:.3f} | ep36={treat[36]:.3f}")
print(f"16th best ep23={sixteenth[23]:.3f}")
