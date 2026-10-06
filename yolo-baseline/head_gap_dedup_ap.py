#!/usr/bin/env python3
"""重複標註對「換頭差值」的影響（head_gap_analysis.py 的補充檢驗）。

head_gap_analysis.py 發現：一對多頭＋NMS 漏掉、一對一頭抓到的 car，幾乎都和另一個同類標準答案框 IoU ≈ 1，
也就是同一台車被標了兩次。NMS 會把同一處的兩個框刪成一個，只配得到其中一個標準答案；一對一頭在訓練時就看過這種
重複標註，會在同一處輸出兩個框。這支程式把「同一張圖、同一類別、IoU > 0.9」的標準答案框只留第一個，
再用正式評估腳本 metrics_from_preds.py 的 coco_ap（pycocotools）重算兩個頭的 AP，看換頭差值剩多少。

為了確認算法和正式結果一致，同時用原始標註重算一次，必須等於 <split>_metrics.json 裡的數字。
需要 pycocotools，在 yolo-baseline 容器內跑（路徑用主機路徑掛載）：
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v /path/to/data:/path/to/data \
    -v <co-detr>/experiments:/workspace/codetr_exp:ro -v <本資料夾>:/workspace/yolo -w /workspace/yolo \
    yolo-baseline:latest python head_gap_dedup_ap.py
輸出：e2e/head_gap/dedup_ap.{json,md}
"""
import contextlib
import copy
import importlib.util
import io
import json
import os

import numpy as np
from pycocotools.coco import COCO

from head_gap_analysis import ANN, CLASSES, OUT, iou, pairs

DUP_IOU = 0.9
spec = importlib.util.spec_from_file_location("mfp", "/workspace/codetr_exp/18th_v18_own_dynamic_bgs_122vids/metrics_from_preds.py")
mfp = importlib.util.module_from_spec(spec); spec.loader.exec_module(mfp)


def coco_from(d):
    c = COCO()
    c.dataset = d
    with contextlib.redirect_stdout(io.StringIO()):
        c.createIndex()
    return c


def dedup(d):
    """同一張圖、同一類別，和前面已保留的框 IoU > DUP_IOU 的標準答案框刪掉。回傳新 dataset 與各類別刪掉的數量。"""
    d = copy.deepcopy(d)
    keep, removed, kept_boxes = [], {c: 0 for c in CLASSES}, {}
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        b = np.array([[x, y, x + w, y + h]], float)
        k = (a["image_id"], a["category_id"])
        prev = kept_boxes.get(k)
        if prev is not None and iou(b, prev).max() > DUP_IOU:
            removed[CLASSES[a["category_id"] - 1]] += 1
            continue
        kept_boxes[k] = b if prev is None else np.vstack([prev, b])
        keep.append(a)
    d["annotations"] = keep
    return d, removed


def ap(coco_gt, pkl):
    with contextlib.redirect_stdout(io.StringIO()):
        return mfp.coco_ap(coco_gt, mfp.load_preds(pkl))


def main():
    raw = {sp: json.load(open(ANN[sp], encoding="utf-8")) for sp in ANN}
    gts, removed = {}, {}
    for sp, d in raw.items():
        dd, removed[sp] = dedup(d)
        gts[sp] = (coco_from(d), coco_from(dd))
    res = {"dup_iou": DUP_IOU, "removed": removed, "rows": []}
    for name, sp, pe, pn in pairs():
        g0, g1 = gts[sp]
        r = dict(name=name, split=sp)
        for head, pkl in [("e2e", pe), ("nms", pn)]:
            orig, dd = ap(g0, pkl), ap(g1, pkl)
            # 原始標註重算必須等於正式結果
            off = json.load(open(pkl.replace(".pkl", "_metrics.json")))["rows"]
            for c in CLASSES:
                assert abs(orig[c]["AP@0.5"] - off[c]["AP50"]) < 5e-4, (pkl, c, orig[c]["AP@0.5"], off[c]["AP50"])
            r[head] = {c: {"orig": orig[c]["AP@0.5"], "dedup": dd[c]["AP@0.5"]} for c in CLASSES}
        res["rows"].append(r)
        print(name, sp, {c: (round(r["e2e"][c]["orig"] - r["nms"][c]["orig"], 3), round(r["e2e"][c]["dedup"] - r["nms"][c]["dedup"], 3)) for c in CLASSES})

    summ = {}
    for c in CLASSES:
        for key in ["orig", "dedup"]:
            for grp, sel in [("test", lambda r: r["split"] != "valid"), ("valid", lambda r: r["split"] == "valid")]:
                rs = [r for r in res["rows"] if sel(r)]
                summ.setdefault(c, {})[f"{grp}_{key}"] = dict(
                    e2e=float(np.mean([r["e2e"][c][key] for r in rs])), nms=float(np.mean([r["nms"][c][key] for r in rs])),
                    diff=float(np.mean([r["e2e"][c][key] - r["nms"][c][key] for r in rs])), n=len(rs))
    res["summary"] = summ
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/dedup_ap.json", "w"), ensure_ascii=False, indent=2)
    md = [f"# 去掉重複標註（同類 IoU > {DUP_IOU}）前後的換頭差值（e2e − NMS，AP@0.5）\n\n",
          "刪掉的標準答案框數：" + "；".join(f"{sp} " + "、".join(f"{c} {n}" for c, n in rm.items() if n) for sp, rm in removed.items()) + "\n\n",
          "| 類別 | test 原始標註 e2e / NMS（差值） | test 去重後 e2e / NMS（差值） | valid 原始 e2e / NMS（差值） | valid 去重後 e2e / NMS（差值） |\n|---|---|---|---|---|\n"]
    for c in CLASSES:
        s = summ[c]
        cell = lambda k: f"{s[k]['e2e']:.3f} / {s[k]['nms']:.3f}（{s[k]['diff']:+.3f}）"
        md.append(f"| {c} | {cell('test_orig')} | {cell('test_dedup')} | {cell('valid_orig')} | {cell('valid_dedup')} |\n")
    md.append("\n> test：6 組（3 seed × dynamic166、static594，各用端到端版選中的 epoch）平均；valid：seed1、seed2 兩組平均。\n")
    open(f"{OUT}/dedup_ap.md", "w", encoding="utf-8").writelines(md)
    print("".join(md))
    print("DEDUP_DONE")


if __name__ == "__main__":
    main()
