#!/usr/bin/env python3
"""端到端（一對一頭）vs 一對多頭＋NMS：差距從哪裡來（《數據彙整（End-to-End Rerun）》討論一節的數據）。

討論一節先從架構設計推出「應該會看到什麼」，這支程式用已經存好的預測檔檢驗那些預測。
只讀既有檔案、不推論，在主機上直接跑：python3 head_gap_analysis.py

比較一律是「同一個權重檔、只換頭」，避免混進選 epoch 的差異：
  test（dynamic166、static594）：每個 seed 用端到端版選中的 epoch（20 / 71 / 43）。
      一對一頭：e2e/seed<N>/test_eval/<split>.pkl
      一對多頭：seed1、seed2 兩版選中的 epoch 相同 → nms/seed<N>/test_eval；seed0 → e2e/head_compare/seed0/nms_ep20（步驟 5b 補跑）
  valid：只有 seed1、seed2 兩版選中同一個 epoch，兩個頭的 valid.pkl 都有；seed0 沒有同 epoch 的一對多頭 valid 預測，不納入。
  valid 每個 epoch 的 AP（epoch_select/valid_ap_per_epoch.json，兩個頭各 3 seed × 100 epoch）：用來看各類別換頭差值的方向。

檢驗項目（編號對應討論一節的預測 P1～P4）：
  P1 各類別訓練實例數 vs 換頭差值（valid 300 組同 epoch 配對、test 三 seed 平均）
  P2 兩個頭都抓到的物體，命中框的信心分數；只有一個頭抓到的物體，那個框的分數
  P3 信心分數下限 0.001 時的最大召回率（IoU ≥ 0.5 與 ≥ 0.3）、按物體大小分組的召回率、靠很近的物體的召回率
  P4 重複框：落在已被命中的標準答案上（IoU ≥ 0.5）、分數 ≥ 0.3 的多餘框數
  P5 每張圖平均輸出幾個框（各類別，分數 ≥ 存檔下限 0.001 的全部框）
  P6 只有一對一頭抓到的物體，有多少是重複標註（和另一個同類標準答案框 IoU > 0.9）；car 靠很近的物體再分成重複標註 / 非重複標註
  P7 分數排序：隨機取一個抓對的框和一個誤報的框，抓對的框分數比較高的機率（等同只看分數排序的 ROC AUC）
  P8 test 各類別的物體來自幾段 clip（檔名去掉 _frame 後的編號即為 clip）
配對規則（與 COCO 相同的貪婪法）：同一張圖、同一類別，預測框依分數由高到低，各自配給 IoU 最高、尚未被配走的標準答案框。
輸出：e2e/head_gap/head_gap.{json,md}
"""
import json
import os
import pickle
from collections import Counter, defaultdict

import numpy as np

ROOT = "/path/to/data/baseline_yolo26l_v17"
COCO = "/path/to/data/coco"
V17 = f"{COCO}/chiayi_v17_own_dynamic_litter_head10_manual/annotations"
V18 = f"{COCO}/chiayi_v18_own_dynamic_bgs_122vids/annotations"
ANN = {"dynamic166": f"{V18}/test_dynamic166.json", "static594": f"{V18}/test_static594.json", "valid": f"{V17}/valid.json"}
OUT = f"{ROOT}/e2e/head_gap"
CLASSES = ["car", "litter", "motorcycle", "person", "garbage"]   # pkl 的 label 0..4 對應 COCO category_id 1..5
DUP_SCORE = 0.3
# 物體大小：標準答案框面積開根號，換算成模型輸入（長邊 1024）上的像素。原圖 2592×1944，縮放 1024/2592。
SIZE_BINS = [(0, 16, "< 16 px"), (16, 32, "16～32 px"), (32, 64, "32～64 px"), (64, 1e9, "≥ 64 px")]
SCALE = 1024 / 2592


def best(head, s):
    return json.load(open(f"{ROOT}/{head}/seed{s}/epoch_select/best_epoch.json"))["best_epoch"]


def iou(a, b):
    """a: (N,4) xyxy, b: (M,4) xyxy → (N,M)"""
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def load_gt(split):
    d = json.load(open(ANN[split], encoding="utf-8"))
    assert {c["id"]: c["name"] for c in d["categories"]} == {i + 1: n for i, n in enumerate(CLASSES)}
    gt = defaultdict(lambda: defaultdict(list))   # img_id → cls → [xyxy]
    for a in d["annotations"]:
        assert not a.get("iscrowd", 0)
        x, y, w, h = a["bbox"]
        gt[a["image_id"]][a["category_id"] - 1].append([x, y, x + w, y + h])
    return {i: {c: np.array(v, dtype=np.float64) for c, v in g.items()} for i, g in gt.items()}, [im["id"] for im in d["images"]]


def match(pred_boxes, pred_scores, gt_boxes, thr):
    """貪婪配對。回傳 gt_hit[j]＝配到第 j 個標準答案的預測框分數（沒配到為 nan），以及每個預測框是否為「重複框」。"""
    gt_hit = np.full(len(gt_boxes), np.nan)
    dup = np.zeros(len(pred_boxes), bool)
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return gt_hit, dup
    ious = iou(pred_boxes, gt_boxes)
    for i in np.argsort(-pred_scores, kind="stable"):
        cand = np.where(np.isnan(gt_hit) & (ious[i] >= thr))[0]
        if len(cand):
            gt_hit[cand[np.argmax(ious[i, cand])]] = pred_scores[i]
        elif (ious[i] >= thr).any():
            dup[i] = True      # 和某個已被命中的標準答案重疊夠多，但那個答案已經被分數更高的框配走
    return gt_hit, dup


def per_gt(pkl, gt, img_ids):
    """對每個標準答案物體，回傳 (類別, 大小, 最近同類物體距離比, 0.5 命中分數, 0.3 命中分數)；另外統計重複框。"""
    preds = {r["img_id"]: r["pred_instances"] for r in pickle.load(open(pkl, "rb"))}
    assert set(preds) == set(img_ids), pkl
    rows, dups = [], Counter()
    for img in img_ids:
        p = preds[img]
        for c, g in gt.get(img, {}).items():
            m = p["labels"] == c
            pb, ps = p["bboxes"][m].astype(np.float64), p["scores"][m].astype(np.float64)
            h5, d5 = match(pb, ps, g, 0.5)
            h3, _ = match(pb, ps, g, 0.3)
            dups[c] += int((d5 & (ps >= DUP_SCORE)).sum())
            size = np.sqrt((g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])) * SCALE
            ctr = np.stack([(g[:, 0] + g[:, 2]) / 2, (g[:, 1] + g[:, 3]) / 2], 1)
            dist = np.sqrt(((ctr[:, None] - ctr[None]) ** 2).sum(-1)); np.fill_diagonal(dist, np.inf)
            near = dist.min(1) * SCALE / np.maximum(size, 1e-9)    # 到最近同類物體的距離 ÷ 自己的大小
            gg = iou(g, g); np.fill_diagonal(gg, 0); twin = gg.max(1)  # 和另一個同類標準答案框的最大 IoU
            for j in range(len(g)):
                rows.append((c, size[j], near[j], h5[j], h3[j], twin[j]))
    return rows, dups


def pred_flags(pkl, gt, img_ids):
    """每個預測框（全部類別、全部影像）是否抓對（IoU ≥ 0.5 貪婪配對）。回傳 cls → [(分數, 是否抓對)]。"""
    preds = {r["img_id"]: r["pred_instances"] for r in pickle.load(open(pkl, "rb"))}
    out = defaultdict(list)
    for img in img_ids:
        p = preds[img]
        for c in range(len(CLASSES)):
            m = p["labels"] == c
            pb, ps = p["bboxes"][m].astype(np.float64), p["scores"][m].astype(np.float64)
            g = gt.get(img, {}).get(c, np.zeros((0, 4)))
            used = np.zeros(len(g), bool)
            for i in np.argsort(-ps, kind="stable"):
                hit = False
                if len(g):
                    io = iou(pb[i:i + 1], g)[0]
                    cand = np.where(~used & (io >= 0.5))[0]
                    if len(cand):
                        used[cand[np.argmax(io[cand])]] = True
                        hit = True
                out[CLASSES[c]].append((ps[i], hit))
    return out


def auc(flags):
    a = np.array(flags, dtype=np.float64)
    tp, fp = np.sort(a[a[:, 1] == 1, 0]), np.sort(a[a[:, 1] == 0, 0])
    if not len(tp) or not len(fp):
        return None
    lo, hi = np.searchsorted(fp, tp, side="left"), np.searchsorted(fp, tp, side="right")
    return float((lo + 0.5 * (hi - lo)).sum() / (len(tp) * len(fp)))


def pairs():
    """同一個權重檔、兩個頭的預測檔配對：(名稱, split, 一對一頭 pkl, 一對多頭 pkl)"""
    out = []
    for s in range(3):
        be, bn = best("e2e", s), best("nms", s)
        nms_dir = f"{ROOT}/nms/seed{s}/test_eval" if be == bn else f"{ROOT}/e2e/head_compare/seed{s}/nms_ep{be}"
        for sp in ["dynamic166", "static594"]:
            out.append((f"seed{s} ep{be}", sp, f"{ROOT}/e2e/seed{s}/test_eval/{sp}.pkl", f"{nms_dir}/{sp}.pkl"))
        if be == bn:
            out.append((f"seed{s} ep{be}", "valid", f"{ROOT}/e2e/seed{s}/test_eval/valid.pkl", f"{ROOT}/nms/seed{s}/test_eval/valid.pkl"))
    for _, _, a, b in out:
        assert os.path.isfile(a) and os.path.isfile(b), (a, b)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"pairs": [], "P1": {}, "P2": {}, "P3": {}, "P4": {}, "P5": {}, "P6": {}, "P7": {}, "P8": {}}

    # P1：訓練實例數；valid 每 epoch 換頭差值（3 seed × 100 epoch 配對）
    tr = json.load(open(f"{V17}/train.json", encoding="utf-8"))
    n_train = Counter(CLASSES[a["category_id"] - 1] for a in tr["annotations"])
    hc = json.load(open(f"{ROOT}/e2e/head_compare/head_compare.json"))
    for c in CLASSES:
        d = []
        for s in range(3):
            e = json.load(open(f"{ROOT}/e2e/seed{s}/epoch_select/valid_ap_per_epoch.json"))
            n = json.load(open(f"{ROOT}/nms/seed{s}/epoch_select/valid_ap_per_epoch.json"))
            d += [e[k][c]["AP@0.5"] - n[k][c]["AP@0.5"] for k in sorted(e, key=int)]
        d = np.array(d); assert len(d) == 300
        test = {sp: float(np.mean([hc["a_same_ckpt_test"][f"seed{s}"][f"{sp}/{c}"]["diff_AP50"] for s in range(3)]))
                for sp in ["dynamic166", "static594"]}
        res["P1"][c] = dict(n_train=n_train[c], valid_diff_mean=float(d.mean()), valid_diff_median=float(np.median(d)),
                            valid_n_e2e_higher=int((d > 0).sum()), test_diff_mean=test)

    # P2～P4：逐物體配對
    gts = {sp: load_gt(sp) for sp in ANN}
    acc = {h: defaultdict(list) for h in ("e2e", "nms")}   # cls → rows（所有配對合併）
    dup_tot = {h: Counter() for h in ("e2e", "nms")}
    n_img = 0
    nbox = {h: np.zeros(5) for h in ("e2e", "nms")}
    flags = {h: defaultdict(list) for h in ("e2e", "nms")}
    for name, sp, pe, pn in pairs():
        gt, ids = gts[sp]
        re_, de = per_gt(pe, gt, ids)
        rn, dn = per_gt(pn, gt, ids)
        assert len(re_) == len(rn)
        for h, p in (("e2e", pe), ("nms", pn)):
            for c, v in pred_flags(p, gt, ids).items():
                flags[h][c] += v
            nbox[h] += np.bincount(np.concatenate([r["pred_instances"]["labels"] for r in pickle.load(open(p, "rb"))]), minlength=5)
        res["pairs"].append(dict(name=name, split=sp, e2e=pe, nms=pn, n_img=len(ids), n_gt=len(re_)))
        n_img += len(ids)
        for r_e, r_n in zip(re_, rn):
            assert r_e[0] == r_n[0] and r_e[1] == r_n[1]
            acc["e2e"][CLASSES[r_e[0]]].append(r_e); acc["nms"][CLASSES[r_n[0]]].append(r_n)
        for c in range(5):
            dup_tot["e2e"][CLASSES[c]] += de[c]; dup_tot["nms"][CLASSES[c]] += dn[c]

    for c in CLASSES:
        E = np.array(acc["e2e"][c], dtype=np.float64); N = np.array(acc["nms"][c], dtype=np.float64)
        he, hn = ~np.isnan(E[:, 3]), ~np.isnan(N[:, 3])
        both, only_n, only_e = he & hn, hn & ~he, he & ~hn
        res["P2"][c] = dict(n_gt=len(E), n_both=int(both.sum()), n_only_nms=int(only_n.sum()), n_only_e2e=int(only_e.sum()),
                            both_score_median_e2e=float(np.median(E[both, 3])) if both.any() else None,
                            both_score_median_nms=float(np.median(N[both, 3])) if both.any() else None,
                            both_e2e_lower_frac=float((E[both, 3] < N[both, 3]).mean()) if both.any() else None,
                            only_nms_score_median=float(np.median(N[only_n, 3])) if only_n.any() else None,
                            only_e2e_score_median=float(np.median(E[only_e, 3])) if only_e.any() else None)
        size_rows = []
        for lo, hi, lab in SIZE_BINS:
            m = (E[:, 1] >= lo) & (E[:, 1] < hi)
            size_rows.append(dict(bin=lab, n=int(m.sum()),
                                  recall_e2e=float(he[m].mean()) if m.any() else None,
                                  recall_nms=float(hn[m].mean()) if m.any() else None))
        near = E[:, 2] < 2.0          # 最近的同類物體，中心距離不到自己大小的 2 倍
        res["P3"][c] = dict(recall50_e2e=float(he.mean()), recall50_nms=float(hn.mean()),
                            recall30_e2e=float((~np.isnan(E[:, 4])).mean()), recall30_nms=float((~np.isnan(N[:, 4])).mean()),
                            by_size=size_rows,
                            near_n=int(near.sum()),
                            near_recall_e2e=float(he[near].mean()) if near.any() else None,
                            near_recall_nms=float(hn[near].mean()) if near.any() else None,
                            far_recall_e2e=float(he[~near].mean()), far_recall_nms=float(hn[~near].mean()))
        res["P6"][c] = dict(n_only_e2e=int(only_e.sum()), n_only_e2e_twin=int((E[only_e, 5] > 0.9).sum()),
                            n_only_nms=int(only_n.sum()), n_only_nms_twin=int((N[only_n, 5] > 0.9).sum()),
                            n_twin_total=int((E[:, 5] > 0.9).sum()))
        oe_nt = only_e & (E[:, 5] <= 0.9)
        res["P6"][c]["only_e2e_not_twin"] = dict(n=int(oe_nt.sum()), score_median=float(np.median(E[oe_nt, 3])) if oe_nt.any() else None)
        for lab, m in [("near_twin", near & (E[:, 5] > 0.9)), ("near_not_twin", near & (E[:, 5] <= 0.9))]:
            res["P6"][c][lab] = dict(n=int(m.sum()), recall_e2e=float(he[m].mean()) if m.any() else None,
                                     recall_nms=float(hn[m].mean()) if m.any() else None)
        fe, fn_ = np.array(flags["e2e"][c]), np.array(flags["nms"][c])
        res["P7"][c] = dict(auc_e2e=auc(flags["e2e"][c]), auc_nms=auc(flags["nms"][c]),
                            n_tp_e2e=int(fe[:, 1].sum()), n_fp_e2e=int((fe[:, 1] == 0).sum()),
                            n_tp_nms=int(fn_[:, 1].sum()), n_fp_nms=int((fn_[:, 1] == 0).sum()))
        res["P4"][c] = dict(dup_e2e=dup_tot["e2e"][c], dup_nms=dup_tot["nms"][c], n_gt=len(E))
    res["n_img_total"] = n_img
    for sp in ["dynamic166", "static594"]:
        d = json.load(open(ANN[sp], encoding="utf-8")); fn = {i["id"]: i["file_name"] for i in d["images"]}
        for c in CLASSES:
            cl = Counter(fn[a["image_id"]].rsplit("_frame", 1)[0] for a in d["annotations"] if CLASSES[a["category_id"] - 1] == c)
            res["P8"][f"{sp}/{c}"] = dict(n_obj=sum(cl.values()), n_clip=len(cl))
    for i, c in enumerate(CLASSES):
        res["P5"][c] = dict(boxes_per_img_e2e=float(nbox["e2e"][i] / n_img), boxes_per_img_nms=float(nbox["nms"][i] / n_img))
    res["note"] = ("合併的配對：test 兩個子集 × 3 seed（各用端到端版選中的 epoch）＋ valid × seed1、seed2。"
                   f"重複框分數門檻 {DUP_SCORE}；大小與距離換算到模型輸入（長邊 1024 px）。")

    json.dump(res, open(f"{OUT}/head_gap.json", "w"), ensure_ascii=False, indent=2)
    f = lambda v: "—" if v is None else f"{v:.3f}"
    md = ["# 換頭差距從哪裡來（同一個權重檔，一對一頭 vs 一對多頭＋NMS）\n\n> " + res["note"] + "\n\n配對清單：\n\n"]
    md += [f"- {p['name']} {p['split']}：{p['n_img']} 張、{p['n_gt']} 個物體\n" for p in res["pairs"]]
    md.append("\n## P1 訓練實例數 vs 換頭差值（e2e − NMS，AP@0.5）\n\n| 類別 | train 實例數 | valid 差值平均（300 組） | valid e2e 較高組數 | test dynamic166 差值 | test static594 差值 |\n|---|---|---|---|---|---|\n")
    for c in sorted(CLASSES, key=lambda c: -res["P1"][c]["n_train"]):
        p = res["P1"][c]
        md.append(f"| {c} | {p['n_train']} | {p['valid_diff_mean']:+.4f} | {p['valid_n_e2e_higher']} / 300 | {p['test_diff_mean']['dynamic166']:+.3f} | {p['test_diff_mean']['static594']:+.3f} |\n")
    md.append("\n## P2 命中框的信心分數（IoU ≥ 0.5）\n\n| 類別 | 物體數 | 兩頭都命中 | 只有 NMS 命中 | 只有 e2e 命中 | 都命中時分數中位數 e2e / NMS | 都命中時 e2e 分數較低的比例 | 只有 NMS 命中時分數中位數 | 只有 e2e 命中時分數中位數 |\n|---|---|---|---|---|---|---|---|---|\n")
    for c in CLASSES:
        p = res["P2"][c]
        md.append(f"| {c} | {p['n_gt']} | {p['n_both']} | {p['n_only_nms']} | {p['n_only_e2e']} | {f(p['both_score_median_e2e'])} / {f(p['both_score_median_nms'])} | {f(p['both_e2e_lower_frac'])} | {f(p['only_nms_score_median'])} | {f(p['only_e2e_score_median'])} |\n")
    md.append("\n## P3 最大召回率（信心分數下限 0.001）\n\n| 類別 | IoU≥0.5 e2e / NMS | IoU≥0.3 e2e / NMS | 靠很近的物體數 | 靠很近 e2e / NMS | 其他 e2e / NMS |\n|---|---|---|---|---|---|\n")
    for c in CLASSES:
        p = res["P3"][c]
        md.append(f"| {c} | {p['recall50_e2e']:.3f} / {p['recall50_nms']:.3f} | {p['recall30_e2e']:.3f} / {p['recall30_nms']:.3f} | {p['near_n']} | {f(p['near_recall_e2e'])} / {f(p['near_recall_nms'])} | {p['far_recall_e2e']:.3f} / {p['far_recall_nms']:.3f} |\n")
    md.append("\n按大小分組（IoU ≥ 0.5 的最大召回率，e2e / NMS，括號內是物體數）：\n\n| 類別 | " + " | ".join(b[2] for b in SIZE_BINS) + " |\n|---|" + "---|" * len(SIZE_BINS) + "\n")
    for c in CLASSES:
        md.append(f"| {c} | " + " | ".join(f"{f(b['recall_e2e'])} / {f(b['recall_nms'])}（{b['n']}）" for b in res["P3"][c]["by_size"]) + " |\n")
    md.append(f"\n## P4 重複框（分數 ≥ {DUP_SCORE}、和已被命中的物體 IoU ≥ 0.5）\n\n| 類別 | 物體數 | e2e 重複框 | NMS 重複框 |\n|---|---|---|---|\n")
    for c in CLASSES:
        p = res["P4"][c]
        md.append(f"| {c} | {p['n_gt']} | {p['dup_e2e']} | {p['dup_nms']} |\n")
    md.append("\n## P5 每張圖平均輸出的框數（存檔下限 0.001 以上）\n\n| 類別 | e2e | NMS |\n|---|---|---|\n")
    for c in CLASSES:
        p = res["P5"][c]
        md.append(f"| {c} | {p['boxes_per_img_e2e']:.2f} | {p['boxes_per_img_nms']:.2f} |\n")
    md.append("\n## P6 只有一個頭抓到的物體中，重複標註（和另一個同類標準答案框 IoU > 0.9）的數量\n\n| 類別 | 重複標註的物體總數 | 只有 e2e 抓到（其中重複標註） | 只有 NMS 抓到（其中重複標註） |\n|---|---|---|---|\n")
    for c in CLASSES:
        p = res["P6"][c]
        md.append(f"| {c} | {p['n_twin_total']} | {p['n_only_e2e']}（{p['n_only_e2e_twin']}） | {p['n_only_nms']}（{p['n_only_nms_twin']}） |\n")
    md.append("\ncar 靠很近的物體再分：" + "；".join(f"{k} {res['P6']['car'][k]['n']} 個，e2e {f(res['P6']['car'][k]['recall_e2e'])} / NMS {f(res['P6']['car'][k]['recall_nms'])}" for k in ("near_twin", "near_not_twin")) + "\n")
    md.append("\n## P7 分數排序（抓對的框分數高於誤報框的機率）\n\n| 類別 | e2e（抓對 / 誤報框數） | NMS（抓對 / 誤報框數） |\n|---|---|---|\n")
    for c in CLASSES:
        p = res["P7"][c]
        md.append(f"| {c} | {f(p['auc_e2e'])}（{p['n_tp_e2e']} / {p['n_fp_e2e']}） | {f(p['auc_nms'])}（{p['n_tp_nms']} / {p['n_fp_nms']}） |\n")
    md.append("\n## P8 test 各類別的物體數與 clip 數\n\n" + "；".join(f"{k} {v['n_obj']} 個 / {v['n_clip']} 段" for k, v in res["P8"].items()) + "\n")
    open(f"{OUT}/head_gap.md", "w", encoding="utf-8").writelines(md)
    print("".join(md))
    print("HEAD_GAP_DONE")


if __name__ == "__main__":
    main()
