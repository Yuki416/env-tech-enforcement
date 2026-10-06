#!/usr/bin/env python3
"""步驟 6：畫 YOLO26l（3 個 seed）與第 16 次 Co-DETR 的 litter PR 曲線，dynamic166 與 static594 各一張。

資料來自 eval_with_threshold.py 存下的 pr_sweep：在 test 上把信心門檻從 0.01 掃到 0.95（每 0.01 一格），
每個門檻用 metrics_from_preds.py 的配對規則算 Precision 與 Recall。
這不是 pycocotools 算 AP 用的那條插值曲線，而是「實際設定某個門檻時會得到的 P/R」，
所以圖上的每個點都對應一個可以部署的門檻設定。
圖上另外標出兩個點：門檻 0.5（圓圈）與 valid 決定的 best-F1 門檻（星號）。

注意：metrics_from_preds.py 的配對是依 IoU 由高到低貪婪配對（不是依分數），所以門檻非常低、放進大量低分框時，
類別錯但 IoU 較高的低分框可能搶走正確答案，Recall 反而下降，曲線尾端會往回折。這只影響 TP/FP/FN，
不影響 pycocotools 算的 AP（AP 依分數排序）。

用法（容器內）：python plot_pr_curve.py --out <dir> <name>=<threshold_eval.json> ...
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cls", default="litter")
    ap.add_argument("runs", nargs="+", help="name=threshold_eval.json")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = [(r.split("=", 1)[0], json.load(open(r.split("=", 1)[1]))) for r in a.runs]

    for split in ["dynamic166", "static594"]:
        plt.figure(figsize=(6.4, 5.2))
        for name, d in runs:
            t = d["test"][split]
            # 門檻太高、一個框都沒有時 Precision 無定義（eval_with_threshold 記成 0）；這些點不畫，否則會從原點拉出假線段
            sw = [x for x in t["pr_sweep"][a.cls] if not (x[1] == 0 and x[2] == 0)]
            is_codetr = "Co-DETR" in name or "16" in name
            col = "#1f4e79" if is_codetr else None
            line, = plt.plot([x[2] for x in sw], [x[1] for x in sw], lw=2.2 if is_codetr else 1.4,
                             color=col, alpha=1 if is_codetr else .8, label=name)
            c = line.get_color()
            p05 = t["at_0.5"][a.cls]; pb = t["at_best_f1"][a.cls]
            plt.scatter([p05["R"]], [p05["P"]], s=55, facecolors="none", edgecolors=c, linewidths=1.6, zorder=3)
            plt.scatter([pb["R"]], [pb["P"]], s=110, marker="*", color=c, zorder=3)
        plt.scatter([], [], s=55, facecolors="none", edgecolors="k", label="threshold 0.5")
        plt.scatter([], [], s=110, marker="*", color="k", label="best-F1 threshold (from valid)")
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.xlim(0, max(0.3, plt.xlim()[1])); plt.ylim(0, 1.02)
        plt.title(f"{a.cls} on {split}: precision vs recall over score thresholds 0.01–0.95")
        plt.grid(alpha=.3); plt.legend(fontsize=8, loc="upper right"); plt.tight_layout()
        p = os.path.join(a.out, f"pr_{a.cls}_{split}.png")
        plt.savefig(p, dpi=140); plt.close()
        print("saved", p)


if __name__ == "__main__":
    main()
