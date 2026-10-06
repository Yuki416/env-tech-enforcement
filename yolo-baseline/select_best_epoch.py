#!/usr/bin/env python3
"""步驟 4：逐 epoch 在 valid 上評估，挑 litter AP@[.5:.95] 最高的 epoch（YOLO_baseline_定案規劃.md）。

規則與第 16 次相同：第 16 次的 save_best=coco/litter_precision 就是 valid 上 litter 類的 COCO AP@[.5:.95]。
這裡用同一支 metrics_from_preds.py 的 coco_ap()（pycocotools）計算，全程不碰 test。

權重從兩版共用的 seed<N>/weights 讀；輸出寫在這一版（YOLO_HEAD）的結果資料夾 <YB>/seed<N>/epoch_select/：
  epoch_select/valid_ap_per_epoch.json   每個 epoch 的 valid 各類 AP@0.5、AP@[.5:.95]
  epoch_select/best_epoch.json           選中的 epoch、分數，以及 Ultralytics best.pt 對應的 epoch
  epoch_select/ap_vs_epoch.png           litter / garbage AP 隨 epoch 變化

用法（容器內，需設 YOLO_HEAD）：python select_best_epoch.py --seed 0
"""
import argparse
import contextlib
import csv
import glob
import io
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from ultralytics import YOLO

from yolo_eval_common import HEAD_LABEL, YB, YB_SHARED, load_metrics_module, predict_to_records

VALID_ANN = "/workspace/datasets/coco/chiayi_v17_own_dynamic_litter_head10_manual/annotations/valid.json"
VALID_IMG = "/workspace/datasets/coco/chiayi_v17_own_dynamic_litter_head10_manual/valid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    mfp = load_metrics_module()
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(VALID_ANN)

    run = f"{YB_SHARED}/seed{a.seed}"                                   # 訓練產物（兩版共用）
    ckpts = sorted(glob.glob(os.path.join(run, "weights", "epoch*.pt")),
                   key=lambda p: int(re.search(r"epoch(\d+)\.pt$", p).group(1)))
    assert len(ckpts) == 100, f"{run}/weights 底下應有 100 個 epoch*.pt，實際 {len(ckpts)}"
    out_dir = f"{YB}/seed{a.seed}/epoch_select"
    os.makedirs(out_dir, exist_ok=True)

    per_epoch = {}
    for p in ckpts:
        ep = int(re.search(r"epoch(\d+)\.pt$", p).group(1)) + 1        # Ultralytics 檔名從 0 起算，這裡轉成 1 起算
        model = YOLO(p)
        records = predict_to_records(model, VALID_ANN, VALID_IMG, device=a.device)
        preds = {r["img_id"]: {k: v for k, v in r["pred_instances"].items()} for r in records}
        with contextlib.redirect_stdout(io.StringIO()):               # COCOeval 的 summarize 很吵
            res = mfp.coco_ap(coco_gt, preds)
        per_epoch[ep] = res
        print(f"epoch {ep:3d}  litter AP@[.5:.95]={res['litter']['AP@0.5:0.95']:.4f}  "
              f"AP@0.5={res['litter']['AP@0.5']:.4f}   garbage AP@[.5:.95]={res['garbage']['AP@0.5:0.95']:.4f}",
              flush=True)
        del model

    json.dump(per_epoch, open(os.path.join(out_dir, "valid_ap_per_epoch.json"), "w"), indent=2)

    # 選最佳：litter AP@[.5:.95] 最高；同分取較早的 epoch
    best_ep = max(per_epoch, key=lambda e: (per_epoch[e]["litter"]["AP@0.5:0.95"], -e))

    # Ultralytics 自己的 best.pt 是第幾個 epoch：results.csv 裡 fitness 最高者
    # （Ultralytics 8.4.171 的 fitness 只看所有類別平均的 mAP@[.5:.95]，權重 [P,R,mAP50,mAP50-95]=[0,0,0,1]，
    #   見 ultralytics/utils/metrics.py 的 Metric.fitness；舊版的 0.1/0.9 加權已不適用）
    ul_best = None
    rc = os.path.join(run, "results.csv")
    if os.path.isfile(rc):
        rows = list(csv.DictReader(open(rc)))
        fit = lambda r: float(r["metrics/mAP50-95(B)"])
        ul_best = int(max(rows, key=fit)["epoch"])

    summary = {
        "rule": "valid 上 litter AP@[.5:.95] 最高（同第 16 次 save_best=coco/litter_precision）",
        "best_epoch": best_ep,
        "head": HEAD_LABEL,
        "best_ckpt": os.path.join(run, "weights", f"epoch{best_ep - 1}.pt"),
        "valid_litter_AP50_95": per_epoch[best_ep]["litter"]["AP@0.5:0.95"],
        "valid_litter_AP50": per_epoch[best_ep]["litter"]["AP@0.5"],
        "ultralytics_best_pt_epoch": ul_best,
        "n_epochs_evaluated": len(per_epoch),
    }
    json.dump(summary, open(os.path.join(out_dir, "best_epoch.json"), "w"), ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    eps = sorted(per_epoch)
    plt.figure(figsize=(8, 4.5))
    for c, col in [("litter", "#c0392b"), ("garbage", "#7d3c98")]:
        plt.plot(eps, [per_epoch[e][c]["AP@0.5:0.95"] for e in eps], color=col, label=f"{c} AP@[.5:.95]")
        plt.plot(eps, [per_epoch[e][c]["AP@0.5"] for e in eps], color=col, ls="--", alpha=.6, label=f"{c} AP@0.5")
    plt.axvline(best_ep, color="k", ls=":", label=f"selected epoch {best_ep}")
    plt.xlabel("epoch"); plt.ylabel("valid AP (metrics_from_preds / pycocotools)")
    plt.title(f"YOLO26l seed{a.seed} ({HEAD_LABEL}): valid AP vs epoch")
    plt.legend(fontsize=8); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ap_vs_epoch.png"), dpi=130)
    print("SELECT_BEST_EPOCH_DONE")


if __name__ == "__main__":
    main()
