#!/usr/bin/env python3
"""步驟 6：信心分數門檻（YOLO_baseline_定案規劃.md 的 D4）。

TP/FP/FN 報兩組：
  (a) 門檻 0.5：與第 16、18 次現有數字同一把尺（metrics_from_preds.py 原本的設定）
  (b) best-F1 門檻：在 valid 上掃門檻，取讓該類 F1 最高的門檻，再套到 test。litter、garbage 各自決定。
      門檻只看 valid 決定，不看 test。

metrics_from_preds.py 把門檻寫死成模組常數 SCORE_THR=0.5；這裡 import 它、改掉常數後呼叫它自己的
tp_fp_fn()，原檔不改，配對規則（greedy、IoU≥0.5、位置中但類別錯記 FN+FP）完全沿用。

同時把 test 上「每個門檻的 Precision / Recall」存下來，供 plot_pr_curve.py 畫 PR 曲線。

用法（容器內）：
  python eval_with_threshold.py --name "YOLO26l seed0" --valid-pkl <valid.pkl> \
      --test-pkl dynamic166=<x.pkl> --test-pkl static594=<y.pkl> --out <result.json>
"""
import argparse
import contextlib
import io
import json

import numpy as np
from pycocotools.coco import COCO

from yolo_eval_common import load_metrics_module

V18 = "/workspace/datasets/coco/chiayi_v18_own_dynamic_bgs_122vids/annotations"
V17 = "/workspace/datasets/coco/chiayi_v17_own_dynamic_litter_head10_manual/annotations"
ANN = {"valid": f"{V17}/valid.json",
       "dynamic166": f"{V18}/test_dynamic166.json",
       "static594": f"{V18}/test_static594.json"}
CLASSES_OF_INTEREST = ["litter", "garbage"]
THRS = [round(t, 2) for t in np.arange(0.01, 0.96, 0.01)]


def sweep(mfp, coco_gt, preds, thrs):
    """回傳 {thr: {class: {TP,FP,FN}}}。"""
    out = {}
    for t in thrs:
        mfp.SCORE_THR = t
        out[t] = mfp.tp_fp_fn(coco_gt, preds)
    mfp.SCORE_THR = 0.5
    return out


def prf(s):
    p = s["TP"] / (s["TP"] + s["FP"]) if s["TP"] + s["FP"] else 0.0
    r = s["TP"] / (s["TP"] + s["FN"]) if s["TP"] + s["FN"] else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--valid-pkl", required=True)
    ap.add_argument("--test-pkl", action="append", required=True, help="split=path，可給多個")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    mfp = load_metrics_module()
    with contextlib.redirect_stdout(io.StringIO()):
        gts = {k: COCO(v) for k, v in ANN.items()}

    # 1) 在 valid 上找每類的 best-F1 門檻（同分取較高門檻，偏向少誤報）
    vs = sweep(mfp, gts["valid"], mfp.load_preds(a.valid_pkl), THRS)
    best_thr = {}
    for c in CLASSES_OF_INTEREST:
        best_thr[c] = max(THRS, key=lambda t: (prf(vs[t][c])[2], t))
    result = {"name": a.name, "best_f1_thr_from_valid": best_thr,
              "valid_f1_at_best": {c: round(prf(vs[best_thr[c]][c])[2], 4) for c in CLASSES_OF_INTEREST},
              "test": {}}

    # 2) test：門檻 0.5 與 best-F1 門檻兩組 TP/FP/FN，加上 PR 掃描
    for item in a.test_pkl:
        split, path = item.split("=", 1)
        preds = mfp.load_preds(path)
        ts = sweep(mfp, gts[split], preds, THRS)
        rec = {"pkl": path, "at_0.5": {}, "at_best_f1": {}, "pr_sweep": {}}
        for c in CLASSES_OF_INTEREST:
            s05 = ts[0.5][c]; sb = ts[best_thr[c]][c]
            rec["at_0.5"][c] = dict(**s05, P=round(prf(s05)[0], 3), R=round(prf(s05)[1], 3), F1=round(prf(s05)[2], 3))
            rec["at_best_f1"][c] = dict(thr=best_thr[c], **sb, P=round(prf(sb)[0], 3), R=round(prf(sb)[1], 3),
                                        F1=round(prf(sb)[2], 3))
            rec["pr_sweep"][c] = [[t, round(prf(ts[t][c])[0], 4), round(prf(ts[t][c])[1], 4)] for t in THRS]
        result["test"][split] = rec
        print(f"\n== {a.name} / {split}")
        for c in CLASSES_OF_INTEREST:
            x, y = rec["at_0.5"][c], rec["at_best_f1"][c]
            print(f"  {c:8s} 門檻0.5  TP/FP/FN={x['TP']}/{x['FP']}/{x['FN']}  P={x['P']:.3f} R={x['R']:.3f}"
                  f"   | best-F1 門檻 {y['thr']:.2f}  TP/FP/FN={y['TP']}/{y['FP']}/{y['FN']}  P={y['P']:.3f} R={y['R']:.3f}")
    print("\nvalid 決定的 best-F1 門檻：", best_thr, " valid F1：", result["valid_f1_at_best"])
    json.dump(result, open(a.out, "w"), ensure_ascii=False, indent=2)
    print("→", a.out)


if __name__ == "__main__":
    main()
