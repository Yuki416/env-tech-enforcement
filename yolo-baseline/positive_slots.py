#!/usr/bin/env python3
"""一對多頭每個物體實際能拿到幾個正樣本（《數據彙整（End-to-End Rerun）》第 13 節、《雙偵測頭》講解頁第 3 節的數據）。

Ultralytics 8.4.171 的 E2ELoss（utils/loss.py 第 1327～1328 行）寫死一對多頭 tal_topk=10、一對一頭 tal_topk=7 + tal_topk2=1。
但 TaskAlignedAssigner 只從「中心點落在答案框內」的格點裡挑（utils/tal.py select_candidates_in_gts），
框內格點不到 10 個時，一對多頭的正樣本就只有那幾個。這支程式數 train 集每個答案框內有幾個格點。

格點：模型輸入長邊 1024 像素（訓練 imgsz=1024），三層特徵圖的步距 8、16、32 像素，格點中心在 (i + 0.5) × 步距。
訓練時有 mosaic 與隨機縮放（scale=0.5，物體大小隨機變成 0.5～1.5 倍），所以同時算縮放 0.5、1.0、1.5 倍，
以及縮放倍率在 0.5～1.5 均勻分布時的平均（21 個等距倍率取平均）。不考慮格點相對物體的平移（實際會隨裁切位置變動）。
另外「同一格點被兩個物體選中時只歸 IoU 高的那個」這條規則會讓擁擠處更少，這裡沒有模擬，所以算出的是上限。
在主機上跑：python3 positive_slots.py
輸出：/path/to/data/baseline_yolo26l_v17/e2e/head_gap/positive_slots.{json,md}
"""
import json
import os
from collections import defaultdict

import numpy as np

TRAIN = "/path/to/data/coco/chiayi_v17_own_dynamic_litter_head10_manual/annotations/train.json"
OUT = "/path/to/data/baseline_yolo26l_v17/e2e/head_gap"
IMGSZ, STRIDES, TOPK = 1024, (8, 16, 32), 10
CLASSES = ["car", "litter", "motorcycle", "person", "garbage"]


def n_inside(x, y, w, h):
    n = 0
    for st in STRIDES:
        c = np.arange(st / 2, IMGSZ * 2, st)   # 縮放 1.5 倍時框可能超出 1024，格點範圍放寬，只數框內的
        n += int(((c > x) & (c < x + w)).sum()) * int(((c > y) & (c < y + h)).sum())
    return n


def main():
    d = json.load(open(TRAIN, encoding="utf-8"))
    im = {i["id"]: i for i in d["images"]}
    per = defaultdict(lambda: defaultdict(list))   # cls → scale → [格點數]
    scales = np.linspace(0.5, 1.5, 21)
    for a in d["annotations"]:
        i = im[a["image_id"]]
        s0 = IMGSZ / max(i["width"], i["height"])
        x, y, w, h = [v * s0 for v in a["bbox"]]
        c = CLASSES[a["category_id"] - 1]
        for s in scales:
            per[c][round(float(s), 2)].append(n_inside(x * s, y * s, w * s, h * s))
    res = {"note": "train 集；框內格點數＝一對多頭正樣本的上限來源；正樣本上限＝min(格點數, 10)。未模擬「一格點歸 IoU 較高物體」規則。",
           "classes": {}}
    for c in CLASSES:
        r = {}
        for key, ss in [("x0.5", [0.5]), ("x1.0", [1.0]), ("x1.5", [1.5]), ("uniform_0.5_1.5", [round(float(s), 2) for s in scales])]:
            v = np.concatenate([np.array(per[c][s]) for s in ss])
            r[key] = dict(n_box=int(len(v) / len(ss)), median_points=float(np.median(v)),
                          mean_pos_o2m=float(np.minimum(v, TOPK).mean()),
                          frac_lt10=float((v < TOPK).mean()), frac_zero=float((v == 0).mean()))
        res["classes"][c] = r
    # litter 細節（不縮放）：框的大小、哪些框三層都沒有格點、8 px 層沒有但靠 16/32 px 層補上的
    W, H, Z, Z8 = [], [], [], []
    for a in d["annotations"]:
        if CLASSES[a["category_id"] - 1] != "litter":
            continue
        i = im[a["image_id"]]
        s0 = IMGSZ / max(i["width"], i["height"])
        x, y, w, h = [v * s0 for v in a["bbox"]]
        c8 = np.arange(4, IMGSZ * 2, 8)
        W.append(w); H.append(h)
        Z.append(n_inside(x, y, w, h) == 0)
        Z8.append(int(((c8 > x) & (c8 < x + w)).sum()) * int(((c8 > y) & (c8 < y + h)).sum()) == 0)
    W, H, Z, Z8 = map(np.array, (W, H, Z, Z8))
    short = np.minimum(W, H)
    res["litter_detail_x1.0"] = dict(
        n_box=int(len(W)), median_w=float(np.median(W)), median_h=float(np.median(H)),
        frac_short_lt8=float((short < 8).mean()), n_zero=int(Z.sum()), frac_zero=float(Z.mean()),
        zero_frac_short_lt8=float((short[Z] < 8).mean()), zero_median_w=float(np.median(W[Z])), zero_median_h=float(np.median(H[Z])),
        n_zero_on_8px_but_rescued=int((Z8 & ~Z).sum()))
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/positive_slots.json", "w"), ensure_ascii=False, indent=2)
    md = ["# 一對多頭每個物體可得的正樣本數（train 集，上限 10）\n\n> " + res["note"] + "\n\n",
          "| 類別 | 框數 | 縮放 | 框內格點數中位數 | 一對多頭正樣本數平均 | 格點不到 10 個的框 | 一個格點都沒有的框 |\n|---|---|---|---|---|---|---|\n"]
    for c in sorted(CLASSES, key=lambda c: -res["classes"][c]["x1.0"]["median_points"]):
        for key, r in res["classes"][c].items():
            md.append(f"| {c} | {r['n_box']} | {key} | {r['median_points']:.0f} | {r['mean_pos_o2m']:.1f} | {r['frac_lt10']:.1%} | {r['frac_zero']:.1%} |\n")
    ld = res["litter_detail_x1.0"]
    md.append(f"\n## litter 細節（不縮放，長邊 1024 px）\n\n框數 {ld['n_box']}；寬、高中位數 {ld['median_w']:.1f} / {ld['median_h']:.1f} px；"
              f"短邊 < 8 px 的框 {ld['frac_short_lt8']:.1%}；三層都沒有格點 {ld['n_zero']} 個（{ld['frac_zero']:.1%}），"
              f"其中短邊 < 8 px 佔 {ld['zero_frac_short_lt8']:.0%}，寬、高中位數 {ld['zero_median_w']:.1f} / {ld['zero_median_h']:.1f} px；"
              f"8 px 層沒有、靠 16 或 32 px 層補上 {ld['n_zero_on_8px_but_rescued']} 個。\n")
    open(f"{OUT}/positive_slots.md", "w", encoding="utf-8").writelines(md)
    print("".join(md))


if __name__ == "__main__":
    main()
