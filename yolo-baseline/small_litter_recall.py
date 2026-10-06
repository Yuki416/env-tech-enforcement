#!/usr/bin/env python3
"""微小 litter 的召回率：YOLO26l（端到端）vs 第 16、18 次 Co-DETR（《數據彙整（End-to-End Rerun）》第 14 節）。

問題：YOLO26 最細的特徵層格點間距 8 px，訓練時只有中心點落在答案框內的格點能當正樣本；Co-DINO 最細一層間距 4 px，
主體用一對一集合配對、不受這條規則限制。如果這個差別要緊，YOLO 的 litter 召回率應該在短邊 < 8 px 時掉得比 Co-DETR 多。

做法：依 litter 答案框的短邊（換算到長邊 1024 px 的模型輸入，原圖 2592×1944，縮放 1024/2592）分組，
算每組被抓到的比例。抓到＝同一張圖、litter 類，預測框依分數由高到低貪婪配給 IoU 最高、還沒被配走的答案框，IoU ≥ 0.5。
  最大召回率：用存下來的全部 litter 框（存檔下限：YOLO 0.001；Co-DETR 見 n_box 欄）。
  前 K 名召回率：每張圖只用分數最高的 K 個 litter 框（K = 10），排除「存的框比較多就比較容易碰到」的影響。
  最佳門檻召回率：只用分數 ≥ 該模型在 valid 上的 best-F1 門檻的框（YOLO 各 seed 的 threshold_eval.json；第 16 次
    codetr16_valid/threshold_eval.json；第 18 次沒有 valid 預測，不算）。
資料：test dynamic166、static594（三個 YOLO seed、第 16 次、第 18 次）；valid（三個 YOLO seed、第 16 次）。
另外統計 YOLO 訓練集裡「三層都沒有格點」的 litter 短邊分布，與 test、valid 各組數量對照（讀 positive_slots.py 的定義重算）。

Co-DETR 的 pkl 存的是 torch tensor，所以在 yolo-baseline 容器內跑（路徑用主機路徑掛載）：
  docker run --rm --user "$(id -u):$(id -g)" -e HOME=/tmp -v /path/to/data:/path/to/data -v <本資料夾>:/workspace/yolo -w /workspace/yolo \
    -e PYTHONUTF8=1 yolo-baseline:latest python small_litter_recall.py
輸出：/path/to/data/baseline_yolo26l_v17/e2e/small_litter/small_litter.{json,md}
"""
import json
import os
import pickle
from collections import defaultdict

import numpy as np

ROOT = "/path/to/data"
YB = f"{ROOT}/baseline_yolo26l_v17"
DT = f"{ROOT}/18th_expt_v18_own_dynamic_bgs_4ch_36e/dual_track_eval"
COCO = f"{ROOT}/coco"
ANN = {"dynamic166": f"{COCO}/chiayi_v18_own_dynamic_bgs_122vids/annotations/test_dynamic166.json",
       "static594": f"{COCO}/chiayi_v18_own_dynamic_bgs_122vids/annotations/test_static594.json",
       "valid": f"{COCO}/chiayi_v17_own_dynamic_litter_head10_manual/annotations/valid.json"}
TRAIN = f"{COCO}/chiayi_v17_own_dynamic_litter_head10_manual/annotations/train.json"
OUT = f"{YB}/e2e/small_litter"
LITTER = 1                     # pkl label（0..4 = car, litter, motorcycle, person, garbage）
SCALE = 1024 / 2592
BINS = [(0, 8, "< 8 px"), (8, 12, "8～12 px"), (12, 16, "12～16 px"), (16, 1e9, "≥ 16 px")]
TOPK = 10


def to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def load_preds(pkl):
    out = {}
    for r in pickle.load(open(pkl, "rb")):
        pi = r["pred_instances"]
        get = (lambda k: pi[k]) if isinstance(pi, dict) else (lambda k: getattr(pi, k))
        lab = to_np(get("labels")).reshape(-1).astype(int)
        m = lab == LITTER
        out[int(r["img_id"])] = (to_np(get("bboxes")).reshape(-1, 4)[m].astype(np.float64), to_np(get("scores")).reshape(-1)[m].astype(np.float64))
    return out


def iou(a, b):
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def load_gt(split):
    d = json.load(open(ANN[split], encoding="utf-8"))
    assert {c["id"]: c["name"] for c in d["categories"]}[LITTER + 1] == "litter"
    assert all((i["width"], i["height"]) == (2592, 1944) for i in d["images"])
    g = defaultdict(list)
    for a in d["annotations"]:
        if a["category_id"] == LITTER + 1:
            x, y, w, h = a["bbox"]
            g[a["image_id"]].append([x, y, x + w, y + h])
    return {k: np.array(v) for k, v in g.items()}, [i["id"] for i in d["images"]]


def hits(preds, gt, ids, thr=None, topk=None):
    """回傳每個 litter 答案框（依影像順序展開）是否被抓到，以及短邊（模型輸入尺度）。"""
    hit, short = [], []
    for img in ids:
        g = gt.get(img)
        if g is None:
            continue
        pb, ps = preds[img]
        if thr is not None:
            k = ps >= thr; pb, ps = pb[k], ps[k]
        o = np.argsort(-ps, kind="stable")
        if topk is not None:
            o = o[:topk]
        pb, ps = pb[o], ps[o]
        used = np.zeros(len(g), bool)
        if len(pb):
            io = iou(pb, g)
            for i in range(len(pb)):
                c = np.where(~used & (io[i] >= 0.5))[0]
                if len(c):
                    used[c[np.argmax(io[i, c])]] = True
        hit += list(used)
        short += list(np.minimum(g[:, 2] - g[:, 0], g[:, 3] - g[:, 1]) * SCALE)
    return np.array(hit), np.array(short)


def models(split):
    out = []
    for s in range(3):
        thr = json.load(open(f"{YB}/e2e/seed{s}/test_eval/threshold_eval.json"))["best_f1_thr_from_valid"]["litter"]
        out.append((f"YOLO26l seed{s}", f"{YB}/e2e/seed{s}/test_eval/{split}.pkl", thr))
    thr16 = json.load(open(f"{YB}/codetr16_valid/threshold_eval.json"))["best_f1_thr_from_valid"]["litter"]
    if split == "valid":
        out.append(("第16次", f"{YB}/codetr16_valid/16th_valid.pkl", thr16))
    else:
        out.append(("第16次", f"{DT}/16th_{split}.pkl", thr16))
        out.append(("第18次", f"{DT}/18th_{split}.pkl", None))
    return out


def train_zero_short():
    """YOLO 訓練集（不縮放）三層都沒有格點的 litter 短邊；定義同 positive_slots.py。"""
    d = json.load(open(TRAIN, encoding="utf-8"))
    im = {i["id"]: i for i in d["images"]}
    short_all, short_zero = [], []
    for a in d["annotations"]:
        if a["category_id"] != LITTER + 1:
            continue
        i = im[a["image_id"]]; s0 = 1024 / max(i["width"], i["height"])
        x, y, w, h = [v * s0 for v in a["bbox"]]
        n = 0
        for st in (8, 16, 32):
            c = np.arange(st / 2, 2048, st)
            n += int(((c > x) & (c < x + w)).sum()) * int(((c > y) & (c < y + h)).sum())
        short_all.append(min(w, h))
        if n == 0:
            short_zero.append(min(w, h))
    return np.array(short_all), np.array(short_zero)


def clip_of_bins(split):
    """各大小組的 litter 來自哪些 clip（檔名去掉 _frame 之後的部分）與各幾個。"""
    d = json.load(open(ANN[split], encoding="utf-8"))
    fn = {i["id"]: i["file_name"].rsplit("_frame", 1)[0] for i in d["images"]}
    out = {lab: defaultdict(int) for _, _, lab in BINS}
    for a in d["annotations"]:
        if a["category_id"] != LITTER + 1:
            continue
        sh = min(a["bbox"][2], a["bbox"][3]) * SCALE
        for lo, hi, lab in BINS:
            if lo <= sh < hi:
                out[lab][fn[a["image_id"]]] += 1
    return {lab: dict(v) for lab, v in out.items()}


def main():
    res = {"bins": [b[2] for b in BINS], "topk": TOPK, "splits": {}, "clips": {}}
    for split in ANN:
        gt, ids = load_gt(split)
        res["clips"][split] = clip_of_bins(split)
        rs = {}
        for name, pkl, thr in models(split):
            P = load_preds(pkl)
            assert set(ids) <= set(P), (pkl, len(set(ids) - set(P)))
            h_all, sh = hits(P, gt, ids)
            h_top, _ = hits(P, gt, ids, topk=TOPK)
            h_thr = hits(P, gt, ids, thr=thr)[0] if thr is not None else None
            nbox = float(np.mean([len(P[i][1]) for i in ids]))
            row = dict(pkl=pkl, best_f1_thr=thr, litter_boxes_per_img=nbox, min_score=float(min((P[i][1].min() for i in ids if len(P[i][1])), default=np.nan)), bins=[])
            for lo, hi, lab in BINS:
                m = (sh >= lo) & (sh < hi)
                row["bins"].append(dict(bin=lab, n=int(m.sum()),
                                        recall_max=float(h_all[m].mean()) if m.any() else None,
                                        recall_top=float(h_top[m].mean()) if m.any() else None,
                                        recall_thr=(float(h_thr[m].mean()) if m.any() else None) if h_thr is not None else None))
            rs[name] = row
        res["splits"][split] = rs
    sa, sz = train_zero_short()
    res["train_litter"] = dict(n=int(len(sa)), n_zero=int(len(sz)), zero_short_max=float(sz.max()),
                               bins=[dict(bin=lab, n=int(((sa >= lo) & (sa < hi)).sum()), n_zero=int(((sz >= lo) & (sz < hi)).sum()))
                                     for lo, hi, lab in BINS])
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/small_litter.json", "w"), ensure_ascii=False, indent=2)
    f = lambda v: "—" if v is None else f"{v:.2f}"
    md = ["# 微小 litter 的召回率（IoU ≥ 0.5；短邊以長邊 1024 px 的模型輸入計）\n\n"]
    for split, rs in res["splits"].items():
        md.append(f"## {split}\n\n| 模型 | 每張圖 litter 框數 | 最低分數 | " + " | ".join(f"{b}（最大 / 前{TOPK} / 最佳門檻）" for b in res["bins"]) + " |\n|---|---|---|" + "---|" * len(BINS) + "\n")
        for name, r in rs.items():
            md.append(f"| {name} | {r['litter_boxes_per_img']:.1f} | {r['min_score']:.4f} | " +
                      " | ".join(f"{f(b['recall_max'])} / {f(b['recall_top'])} / {f(b['recall_thr'])}（{b['n']}）" for b in r["bins"]) + " |\n")
        md.append("\n")
    for split, cb in res["clips"].items():
        md.append(f"## {split} 各大小組的 clip 組成\n\n" + "".join(f"- {lab}：{len(v)} 段 clip，" + "、".join(f"{k}（{n}）" for k, n in sorted(v.items(), key=lambda x: -x[1])) + "\n" for lab, v in cb.items()) + "\n")
    t = res["train_litter"]
    md.append(f"## YOLO 訓練集 litter（不縮放）\n\n共 {t['n']} 個，三層都沒有格點 {t['n_zero']} 個，這些框的短邊最大 {t['zero_short_max']:.1f} px。各組：" +
              "；".join(f"{b['bin']} {b['n']} 個（其中沒有格點 {b['n_zero']}）" for b in t["bins"]) + "\n")
    open(f"{OUT}/small_litter.md", "w", encoding="utf-8").writelines(md)
    print("".join(md))


if __name__ == "__main__":
    main()
