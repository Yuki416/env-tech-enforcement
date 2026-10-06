"""逐框配對：照 metrics_from_preds.py 的 tp_fp_fn() 規則，但回傳「每個框」的判定，而不只是總數。

規則（與 metrics_from_preds.tp_fp_fn 完全相同）：
  1. 只看信心分數 ≥ 門檻的預測框
  2. 把所有「正確答案框 × 預測框」中 IoU ≥ 0.5 的配對，依 IoU 由高到低排序，貪婪地一對一配對
  3. 配上且類別相同 → 該正確答案框 TP、該預測框 TP
     配上但類別不同 → 該正確答案框 FN、該預測框 FP（位置對了但類別錯）
  4. 沒配上的正確答案框 → FN；沒配上的預測框 → FP

使用時一律搭配 check_against_reference() 核對：逐框判定的加總必須等於原腳本 tp_fp_fn() 的結果，否則停止。
"""
CLASSES = ["car", "litter", "motorcycle", "person", "garbage"]
IOU_THR = 0.5


def iou_xyxy(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match_image(anns, pred, score_thr):
    """anns：COCO 標註 list；pred：{"bboxes","scores","labels"}。
    回傳 gts：[(類別, xyxy, "TP"/"FN")]、dets：[(類別, 分數, xyxy, "TP"/"FP")]（只含分數 ≥ 門檻者）。"""
    gtb = [[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]] for a in anns]
    gtc = [a["category_id"] - 1 for a in anns]
    keep = [i for i, s in enumerate(pred["scores"]) if s >= score_thr]
    pb = [list(map(float, pred["bboxes"][i])) for i in keep]
    pc = [int(pred["labels"][i]) for i in keep]
    ps = [float(pred["scores"][i]) for i in keep]
    pairs = []
    for i, g in enumerate(gtb):
        for j, d in enumerate(pb):
            v = iou_xyxy(g, d)
            if v >= IOU_THR:
                pairs.append((v, i, j))
    pairs.sort(reverse=True)
    gst = ["FN"] * len(gtb)
    dst = ["FP"] * len(pb)
    mg, mp = set(), set()
    for v, i, j in pairs:
        if i in mg or j in mp:
            continue
        mg.add(i); mp.add(j)
        if gtc[i] == pc[j]:
            gst[i] = "TP"; dst[j] = "TP"
        # 類別不同：維持 gst=FN、dst=FP
    gts = [(CLASSES[c], b, s) for c, b, s in zip(gtc, gtb, gst)]
    dets = [(CLASSES[c], sc, b, s) for c, sc, b, s in zip(pc, ps, pb, dst)]
    return gts, dets


def counts_from_matches(per_image):
    """per_image：{img_id: (gts, dets)} → {類別: {TP, FP, FN}}，算法與 tp_fp_fn 相同。"""
    st = {c: {"TP": 0, "FP": 0, "FN": 0} for c in CLASSES}
    for gts, dets in per_image.values():
        for c, _, s in gts:
            if s == "TP":
                st[c]["TP"] += 1
            else:
                st[c]["FN"] += 1
        for c, _, _, s in dets:
            if s == "FP":
                st[c]["FP"] += 1
    return st


def check_against_reference(mfp, coco_gt, preds, score_thr):
    """逐框配對的加總必須等於 metrics_from_preds.tp_fp_fn() 的結果。回傳 per_image 配對結果。"""
    per_image = {}
    for img_id in coco_gt.getImgIds():
        p = preds.get(img_id, {"bboxes": [], "scores": [], "labels": []})
        per_image[img_id] = match_image(coco_gt.imgToAnns.get(img_id, []), p, score_thr)
    mine = counts_from_matches(per_image)
    old = mfp.SCORE_THR
    mfp.SCORE_THR = score_thr
    ref = mfp.tp_fp_fn(coco_gt, preds)
    mfp.SCORE_THR = old
    assert mine == ref, f"逐框配對與 tp_fp_fn 不一致（門檻 {score_thr}）：\n{mine}\n{ref}"
    return per_image, mine
