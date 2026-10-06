"""任務4：畫第18次訓練 conv1 第4通道（changemap 投影）的 L2 norm 隨訓練的變化。
目的：客觀佐證「第4通道有沒有被學動」——有動=模型吃了 changemap；幾乎不動=BGS 訊號沒被用到。
只陳述事實，不推論成敗。讀 train.log 的 conv1_ch4_norm / conv1_rgb_norm。"""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "/path/to/data/18th_expt_v18_own_dynamic_bgs_4ch_36e/train.log"
OUT = "/path/to/data/18th_expt_v18_own_dynamic_bgs_4ch_36e/conv1_ch4_norm.png"

pat = re.compile(r"Epoch\(train\) +\[(\d+)\]\[(\d+)/(\d+)\].*?conv1_ch4_norm: ([\d.]+) +conv1_rgb_norm: ([\d.]+)")
step, ch4, rgb, epx = [], [], [], []
for line in Path(LOG).read_text(errors="ignore").splitlines():
    m = pat.search(line)
    if m:
        ep, it, tot = int(m.group(1)), int(m.group(2)), int(m.group(3))
        step.append((ep - 1) + it / tot)
        ch4.append(float(m.group(4))); rgb.append(float(m.group(5)))
if not step:
    raise SystemExit("未解析到 conv1_ch4_norm")

fig, ax1 = plt.subplots(figsize=(11, 6))
ax1.plot(step, ch4, "-", color="#2ca02c", lw=1.6, label="conv1 ch4 (changemap) L2 norm")
ax1.plot(step, rgb, "-", color="#7f7f7f", lw=1.2, alpha=.7, label="conv1 rgb L2 norm")
ax1.set_xlabel("epoch"); ax1.set_ylabel("conv1 channel L2 norm")
ax1.set_title("18th (v18 BGS 4ch) - conv1 ch4(changemap) L2 norm vs training")
for d in (28, 34):
    ax1.axvline(d, color="gray", ls=":", lw=1)
ax2 = ax1.twinx()
ratio = [c / r for c, r in zip(ch4, rgb)]
ax2.plot(step, ratio, "-", color="#1f77b4", lw=1.0, alpha=.5, label="ratio ch4/rgb")
ax2.set_ylabel("ratio ch4 / rgb", color="#1f77b4")
l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, la1 + la2, loc="upper right", fontsize=9)
ax1.grid(alpha=.3)
plt.tight_layout(); plt.savefig(OUT, dpi=130)
print(f"saved: {OUT}")
print(f"ch4 norm: init {ch4[0]:.3f} → final {ch4[-1]:.3f}  (Δ {ch4[-1]-ch4[0]:+.3f}, {100*(ch4[-1]-ch4[0])/ch4[0]:+.1f}%)")
print(f"ch4 norm: min {min(ch4):.3f}  max {max(ch4):.3f}")
print(f"ratio ch4/rgb: init {ratio[0]:.3f} → final {ratio[-1]:.3f}")
