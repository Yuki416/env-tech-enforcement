#!/usr/bin/env python3
"""離線指標：讀 tools/test.py --out 的預測 pkl + GT ann → 每類 AP@0.5、AP@[.5:.95]、TP/FP/FN。
口徑與 error_analysis_v9 一致（score_thr=0.5、IoU=0.5；AP 由 pycocotools）。跨模型同腳本可比。
（複製自 18th metrics_from_preds.py，ablation 自成一套。）

用法：python metrics_from_preds.py --preds x.pkl --ann test_static594.json --tag "statictail static594" [--json-out out.json]
"""
import argparse, pickle, json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CLASSES = ["car", "litter", "motorcycle", "person", "garbage"]
SCORE_THR, IOU_THR = 0.5, 0.5

def to_np(x):
    try:
        return x.cpu().numpy()
    except AttributeError:
        return np.asarray(x)

def load_preds(pkl):
    data = pickle.load(open(pkl, "rb"))
    out = {}
    for r in data:
        pi = r["pred_instances"]
        get = (lambda k: pi[k]) if isinstance(pi, dict) else (lambda k: getattr(pi, k))
        out[r["img_id"]] = dict(
            bboxes=to_np(get("bboxes")).reshape(-1, 4),
            scores=to_np(get("scores")).reshape(-1),
            labels=to_np(get("labels")).reshape(-1).astype(int),
        )
    return out

def iou_xyxy(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def coco_ap(coco_gt, preds, prefilter=None):
    # prefilter=None → 全清單標準 COCO AP；prefilter=0.5 → 先砍 score<0.5 再進 COCOeval，
    # 複製 error_analysis_v9 的「截斷 AP」定義（讓 4ch 的 18th 與變因①/④ 同刻度）。
    dets = []
    for img_id, p in preds.items():
        for bb, sc, lb in zip(p["bboxes"], p["scores"], p["labels"]):
            if prefilter is not None and float(sc) < prefilter:
                continue
            x1, y1, x2, y2 = bb
            dets.append(dict(image_id=int(img_id), category_id=int(lb) + 1,
                             bbox=[float(x1), float(y1), float(x2-x1), float(y2-y1)], score=float(sc)))
    if not dets:
        return {c: {"AP@0.5": 0.0, "AP@0.5:0.95": 0.0} for c in CLASSES}
    coco_dt = coco_gt.loadRes(dets)
    res = {}
    for i, c in enumerate(CLASSES):
        e = COCOeval(coco_gt, coco_dt, "bbox"); e.params.catIds = [i + 1]
        e.evaluate(); e.accumulate(); e.summarize()
        res[c] = {"AP@0.5": float(e.stats[1]), "AP@0.5:0.95": float(e.stats[0])}
    return res

def tp_fp_fn(coco_gt, preds):
    id2anns = {im: coco_gt.imgToAnns.get(im, []) for im in coco_gt.getImgIds()}
    stat = {c: {"TP": 0, "FP": 0, "FN": 0} for c in CLASSES}
    for img_id, anns in id2anns.items():
        gtb = [[a["bbox"][0], a["bbox"][1], a["bbox"][0]+a["bbox"][2], a["bbox"][1]+a["bbox"][3]] for a in anns]
        gtc = [a["category_id"] - 1 for a in anns]
        p = preds.get(img_id, dict(bboxes=np.zeros((0,4)), scores=np.zeros(0), labels=np.zeros(0, int)))
        keep = p["scores"] >= SCORE_THR
        pb, pc = p["bboxes"][keep], p["labels"][keep]
        pairs = []
        for i, g in enumerate(gtb):
            for j, d in enumerate(pb):
                if iou_xyxy(g, d) >= IOU_THR:
                    pairs.append((iou_xyxy(g, d), i, j))
        pairs.sort(reverse=True)
        mg, mp = set(), set()
        for v, i, j in pairs:
            if i in mg or j in mp:
                continue
            mg.add(i); mp.add(j)
            if gtc[i] == pc[j]:
                stat[CLASSES[gtc[i]]]["TP"] += 1
            else:
                stat[CLASSES[gtc[i]]]["FN"] += 1
                stat[CLASSES[pc[j]]]["FP"] += 1
        for i in range(len(gtb)):
            if i not in mg:
                stat[CLASSES[gtc[i]]]["FN"] += 1
        for j in range(len(pb)):
            if j not in mp:
                stat[CLASSES[pc[j]]]["FP"] += 1
    return stat

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--json-out", default="")
    ap.add_argument("--score-prefilter", type=float, default=None,
                    help="算 AP 前先砍 score< 此值的框（=0.5 複製 error_analysis_v9 截斷 AP）；不給=全清單標準 AP")
    a = ap.parse_args()
    coco_gt = COCO(a.ann)
    preds = load_preds(a.preds)
    apr = coco_ap(coco_gt, preds, prefilter=a.score_prefilter)
    tff = tp_fp_fn(coco_gt, preds)
    pf = "none" if a.score_prefilter is None else f"{a.score_prefilter}"
    print(f"\n===== {a.tag} =====  (score{SCORE_THR}, IoU{IOU_THR}, AP_prefilter={pf}; ann={a.ann.split('/')[-1]})")
    print(f"{'class':11s} {'AP@0.5':>7s} {'AP.5:.95':>8s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'Prec':>6s} {'Rec':>6s}")
    rows = {}
    for c in CLASSES:
        s = tff[c]; ap50 = apr[c]["AP@0.5"]; apc = apr[c]["AP@0.5:0.95"]
        prec = s["TP"]/(s["TP"]+s["FP"]) if s["TP"]+s["FP"] else 0.0
        rec = s["TP"]/(s["TP"]+s["FN"]) if s["TP"]+s["FN"] else 0.0
        print(f"{c:11s} {ap50:7.3f} {apc:8.3f} {s['TP']:5d} {s['FP']:5d} {s['FN']:5d} {prec:6.3f} {rec:6.3f}")
        rows[c] = dict(AP50=round(ap50,3), AP=round(apc,3), **s, Prec=round(prec,3), Rec=round(rec,3))
    if a.json_out:
        json.dump(dict(tag=a.tag, ann=a.ann, rows=rows), open(a.json_out, "w"), ensure_ascii=False, indent=2)
        print(f"→ {a.json_out}")

if __name__ == "__main__":
    main()
