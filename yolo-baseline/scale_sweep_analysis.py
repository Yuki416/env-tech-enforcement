#!/usr/bin/env python3
"""縮放實驗的分析（《數據彙整（End-to-End Rerun）》第 14 節）。

讀 scale_sweep_yolo.py 與 scale_sweep_codetr16.sh 的預測檔（valid，輸入長邊 512 / 768 / 1024 / 1536），
對每個模型、每種尺寸算 litter 的：
  AP@0.5          正式評分程式 metrics_from_preds.py 的 coco_ap（pycocotools）
  最大召回率       存下的全部 litter 框，IoU ≥ 0.5 貪婪配對（配對規則同 small_litter_recall.py）
  前 10 名召回率   每張圖只用分數最高的 10 個 litter 框，排除「存的框比較多」的影響
  按「原圖上的短邊」分組的最大召回率：分組固定用輸入 1024 時的短邊（< 8、8～12、12～16、≥ 16 px），
    同一組 litter 在不同輸入尺寸下的大小 = 該值 × 尺寸 / 1024。
另外記錄每種尺寸下，litter 短邊的中位數（模型輸入尺度），以及短邊 < 8 px 的比例。
1024 的結果另外與正式預測檔（e2e/seed<N>/test_eval/valid.pkl、codetr16_valid/16th_valid.pkl）比對 AP，確認重現。

需要 torch 與 pycocotools，在 yolo-baseline 容器內跑：
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v /path/to/data:/path/to/data -v <本資料夾>:/workspace/yolo \
    -v <co-detr>/experiments:/workspace/codetr_exp:ro -w /workspace/yolo -e PYTHONUTF8=1 \
    yolo-baseline:latest python scale_sweep_analysis.py
輸出：/path/to/data/baseline_yolo26l_v17/e2e/scale_sweep/scale_sweep.{json,md}
"""
import contextlib
import importlib.util
import io
import json

import numpy as np
from pycocotools.coco import COCO

from small_litter_recall import BINS, YB, hits, load_gt, load_preds

SW = f"{YB}/e2e/scale_sweep"
SIZES = [512, 768, 1024, 1536]
VALID = "/path/to/data/coco/chiayi_v17_own_dynamic_litter_head10_manual/annotations/valid.json"
spec = importlib.util.spec_from_file_location("mfp", "/workspace/codetr_exp/18th_v18_own_dynamic_bgs_122vids/metrics_from_preds.py")
mfp = importlib.util.module_from_spec(spec); spec.loader.exec_module(mfp)


def coco_ap(coco_gt, pkl):
    with contextlib.redirect_stdout(io.StringIO()):
        return mfp.coco_ap(coco_gt, mfp.load_preds(pkl))


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        cg = COCO(VALID)
    gt, ids = load_gt("valid")
    models = {f"YOLO26l seed{s}": (lambda sz, s=s: f"{SW}/yolo_seed{s}_{sz}/valid.pkl") for s in range(3)}
    models["第16次"] = lambda sz: f"{SW}/16th_{sz}/valid.pkl"
    official = {f"YOLO26l seed{s}": f"{YB}/e2e/seed{s}/test_eval/valid.pkl" for s in range(3)}
    official["第16次"] = f"{YB}/codetr16_valid/16th_valid.pkl"

    res = {"sizes": SIZES, "bins_at_1024": [b[2] for b in BINS], "models": {}, "litter_size": {}, "repro_1024": {}}
    _, short1024 = hits({i: (np.zeros((0, 4)), np.zeros(0)) for i in ids}, gt, ids)   # 只為取得短邊
    for sz in SIZES:
        sh = short1024 * sz / 1024
        res["litter_size"][sz] = dict(median_short=float(np.median(sh)), frac_lt8=float((sh < 8).mean()))
    for name, path in models.items():
        rows = {}
        for sz in SIZES:
            P = load_preds(path(sz))
            h_all, short = hits(P, gt, ids)
            h_top, _ = hits(P, gt, ids, topk=10)
            assert np.allclose(short, short1024)
            ap = coco_ap(cg, path(sz))["litter"]["AP@0.5"]
            rows[sz] = dict(ap50=float(ap), recall_max=float(h_all.mean()), recall_top10=float(h_top.mean()),
                            boxes_per_img=float(np.mean([len(P[i][1]) for i in ids])),
                            by_bin=[dict(bin=lab, n=int(((short >= lo) & (short < hi)).sum()),
                                         recall_max=float(h_all[(short >= lo) & (short < hi)].mean()))
                                    for lo, hi, lab in BINS])
        res["models"][name] = rows
        a_new, a_off = rows[1024]["ap50"], coco_ap(cg, official[name])["litter"]["AP@0.5"]
        res["repro_1024"][name] = dict(sweep=a_new, official=float(a_off), diff=float(a_new - a_off))

    json.dump(res, open(f"{SW}/scale_sweep.json", "w"), ensure_ascii=False, indent=2)
    md = ["# 縮放實驗：valid 上 litter 隨輸入尺寸的變化\n\n",
          "| 輸入長邊 | litter 短邊中位數（模型輸入尺度） | 短邊 < 8 px 的比例 |\n|---|---|---|\n"]
    md += [f"| {sz} | {v['median_short']:.1f} px | {v['frac_lt8']:.1%} |\n" for sz, v in res["litter_size"].items()]
    md.append("\n| 模型 | 輸入長邊 | litter AP@0.5 | 最大召回率 | 前 10 名召回率 | 每張圖 litter 框數 | " + " | ".join(f"1024 時 {b}" for b in res["bins_at_1024"]) + " |\n|---|---|---|---|---|---|" + "---|" * len(BINS) + "\n")
    for name, rows in res["models"].items():
        for sz, r in rows.items():
            md.append(f"| {name} | {sz} | {r['ap50']:.3f} | {r['recall_max']:.3f} | {r['recall_top10']:.3f} | {r['boxes_per_img']:.1f} | " +
                      " | ".join(f"{b['recall_max']:.2f}（{b['n']}）" for b in r["by_bin"]) + " |\n")
    md.append("\n1024 重現檢查（litter AP@0.5，縮放實驗 vs 正式預測檔）：" +
              "；".join(f"{k} {v['sweep']:.4f} vs {v['official']:.4f}" for k, v in res["repro_1024"].items()) + "\n")
    open(f"{SW}/scale_sweep.md", "w", encoding="utf-8").writelines(md)
    print("".join(md))


if __name__ == "__main__":
    main()
