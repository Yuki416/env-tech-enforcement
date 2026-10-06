"""畫第 15 次訓練（v16 動態子集）的 val litter / garbage AP vs epoch，並疊上第 13 次當對照。

為什麼可以疊：**兩次的 valid 是完全相同的 488 張 / 14 clips**（已逐檔名驗證），
同一把尺，可直接比較。第 13 次的曲線畫成虛線＋低透明度當背景參照。

編碼（不靠顏色單獨辨識）：
  - 顏色 = 類別（litter 紅 / garbage 藍）—— 沿用第 13 次 plot_ap_vs_epoch_13th.py 的色票，
    已通過 dataviz 色票驗證（protan ΔE 21.1、normal ΔE 31.7，皆遠高於門檻 8.0 / 15.0）。
  - 線型 = 實驗次數（實線 = 第 15 次 / 虛線 = 第 13 次）
  → 即使色盲或黑白列印，仍可用線型區分兩次實驗。

單一 Y 軸（AP），不做雙軸。

用法：python3 plot_ap_vs_epoch_15th.py
訓練中可隨時跑（標題會標示目前進度）；訓練跑完後重跑一次即為正式版（同檔名覆蓋）。
"""
import re
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_15TH = "/path/to/data/15th_expt_v16_own_dynamic_121vids_36e/train.log"
LOG_13TH = "/path/to/data/13th_expt_v14_own_dynamic_36e/train.log"
OUT = "/path/to/data/15th_expt_v16_own_dynamic_121vids_36e/ap_vs_epoch.png"
MAX_EPOCHS = 36
LR_DROPS = (28, 34)  # milestones=[28,34]，兩次相同

C_LITTER, C_GARBAGE = "#d62728", "#1f77b4"
INK, MUTED = "#1a1a19", "#8a8a85"

pat = re.compile(
    r"Epoch\(val\) \[(\d+)\]\[\d+/\d+\].*?litter_precision: ([\d.]+).*?garbage_precision: ([\d.]+)")


def parse(log):
    eps, lit, gar = [], [], []
    p = Path(log)
    if not p.exists():
        return eps, lit, gar
    for line in p.read_text(errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            eps.append(int(m.group(1)))
            lit.append(float(m.group(2)))
            gar.append(float(m.group(3)))
    return eps, lit, gar


e15, l15, g15 = parse(LOG_15TH)
e13, l13, g13 = parse(LOG_13TH)
if not e15:
    raise SystemExit("第 15 次 log 還沒有任何 val 結果")

done = len(e15) >= MAX_EPOCHS
status = f"complete, {MAX_EPOCHS} epochs" if done else f"IN PROGRESS - up to ep{max(e15)}/{MAX_EPOCHS}"

fig, ax = plt.subplots(figsize=(11, 6))

# ── 第 13 次（對照，虛線退到背景）──────────────────────────────────────────
if e13:
    ax.plot(e13, l13, "--", color=C_LITTER, lw=1.5, alpha=0.45,
            label="13th v14 litter (44 clips, baseline)")
    ax.plot(e13, g13, "--", color=C_GARBAGE, lw=1.5, alpha=0.45,
            label="13th v14 garbage (44 clips, baseline)")

# ── 第 15 次（主角，實線＋標記）────────────────────────────────────────────
ax.plot(e15, l15, "-o", color=C_LITTER, lw=2, markersize=5,
        label="15th v16 litter (99 clips)")
ax.plot(e15, g15, "-o", color=C_GARBAGE, lw=2, markersize=5,
        label="15th v16 garbage (99 clips)")

# ── LR drop 標線（recessive）───────────────────────────────────────────────
for d in LR_DROPS:
    ax.axvline(d, color=MUTED, ls=":", lw=1, zorder=0)
    ax.text(d, 0.985, f"LR drop ep{d}", color=MUTED, fontsize=8, rotation=90,
            va="top", ha="right", transform=ax.get_xaxis_transform())

# ── 只直接標註第 15 次的 best（不是每個點都標數字）─────────────────────────
for vals, color, name in ((l15, C_LITTER, "litter"), (g15, C_GARBAGE, "garbage")):
    i = max(range(len(vals)), key=lambda k: vals[k])
    ax.scatter([e15[i]], [vals[i]], s=110, facecolors="none",
               edgecolors=color, linewidths=2, zorder=5)
    ax.annotate(f"15th {name} best\nep{e15[i]}  AP={vals[i]:.3f}",
                (e15[i], vals[i]), textcoords="offset points", xytext=(9, 7),
                color=INK, fontsize=9)

ax.set_title(
    f"15th (v16, 99 clips) vs 13th (v14, 44 clips) - val litter/garbage AP@[.50:.95]\n"
    f"identical 488-frame valid set  |  {status}",
    color=INK, fontsize=11)
ax.set_xlabel("epoch", color=INK)
ax.set_ylabel("AP@[0.50:0.95] (COCO)", color=INK)
top = max(l15 + g15 + (l13 + g13 if e13 else [0])) * 1.22
ax.set_ylim(0, top)
ax.set_xlim(0.5, MAX_EPOCHS + 0.5)
ax.grid(alpha=0.25, zorder=0)
ax.tick_params(colors=MUTED)
for s in ax.spines.values():
    s.set_color(MUTED)
ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

fig.tight_layout()
fig.savefig(OUT, dpi=130)
print(f"saved: {OUT}")
print(f"status: {status}")
for nm, e, l, g in (("15th", e15, l15, g15), ("13th", e13, l13, g13)):
    if not e:
        continue
    li = max(range(len(l)), key=lambda k: l[k])
    gi = max(range(len(g)), key=lambda k: g[k])
    print(f"  {nm}: litter best ep{e[li]} AP={l[li]:.4f} | garbage best ep{e[gi]} AP={g[gi]:.4f}"
          f" | epochs parsed={len(e)}")
