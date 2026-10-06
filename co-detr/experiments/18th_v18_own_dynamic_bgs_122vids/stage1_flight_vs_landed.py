#!/usr/bin/env python3
"""Stage 1：一幀被送進模型之後，飛行中 vs 落地後的偵測率各是多少？

背景（見 codetr_service/ 的 stride 討論）：
  stride 35 之下只有 12.8% 的飛行段會被取樣到。但落地後 changemap 還會亮 6 次 apply，
  所以 stride 35 仍有「落地後補抓」的機會。這支腳本量那個機會值多少。

方法
  ① 用 GT 框的逐幀位移把 litter/garbage 框分成 flight / landed
     （門檻與 Stage 0 相同：位移 > max(8px, 0.3×框寬) 視為飛行）
  ② 讀 tools/test.py dump 的預測 pkl，對每個 GT 框判斷有沒有被命中
     （同類、IoU≥0.5、score≥門檻），一個預測框只能認領一個 GT
  ③ 分別統計 flight / landed 的 recall，並拆 score 門檻看穩定性

輸出純數字，不下判讀。
"""
import argparse, pickle, json, re, collections
import numpy as np

CLASSES = ["car", "litter", "motorcycle", "person", "garbage"]
LITTER_CID, GARBAGE_CID = 2, 5          # COCO category_id
NAME = {LITTER_CID: "litter", GARBAGE_CID: "garbage"}

def to_np(x):
    try:    return x.cpu().numpy()
    except AttributeError: return np.asarray(x)

def load_preds(pkl):
    out = {}
    for r in pickle.load(open(pkl, "rb")):
        pi = r["pred_instances"]
        get = (lambda k: pi[k]) if isinstance(pi, dict) else (lambda k: getattr(pi, k))
        out[r["img_id"]] = dict(
            bboxes=to_np(get("bboxes")).reshape(-1, 4),
            scores=to_np(get("scores")).reshape(-1),
            labels=to_np(get("labels")).reshape(-1).astype(int))
    return out

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1)
    inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

def clip_key(fn):
    g = re.search(r"(\d{14})_(?:(\d{2})h)?(\d{2})m(\d{2})s", fn)
    h = int(g.group(2)) if g.group(2) else 0
    return (g.group(1), h*3600 + int(g.group(3))*60 + int(g.group(4)))

def frame_no(fn):
    return int(re.search(r"_frame_(\d+)", fn).group(1))

def classify_flight(ann_path):
    """回傳 {(img_id, ann_id): 'flight'|'landed'}，以及每個 clip 的飛行段長度。"""
    d = json.load(open(ann_path))
    id2fn = {im["id"]: im["file_name"] for im in d["images"]}
    # clip -> frame_no -> [(ann_id, img_id, bbox, cat)]
    clips = collections.defaultdict(lambda: collections.defaultdict(list))
    for a in d["annotations"]:
        if a["category_id"] not in (LITTER_CID, GARBAGE_CID):
            continue
        fn = id2fn[a["image_id"]]
        clips[clip_key(fn)][frame_no(fn)].append(
            (a["id"], a["image_id"], a["bbox"], a["category_id"]))

    tag = {}
    for ck, frames in clips.items():
        nos = sorted(frames)
        widths = [b[2] for fr in frames.values() for (_, _, b, _) in fr]
        thr = max(8.0, 0.3*float(np.median(widths)))
        prev_centers = None
        for n in nos:
            items = frames[n]
            centers = [(b[0]+b[2]/2, b[1]+b[3]/2) for (_, _, b, _) in items]
            for (aid, iid, b, cat), c in zip(items, centers):
                if prev_centers is None:
                    tag[(iid, aid)] = "landed"      # clip 第一幀無從判斷，保守算落地
                else:
                    dmin = min(abs(c[0]-p[0]) + abs(c[1]-p[1]) for p in prev_centers)
                    tag[(iid, aid)] = "flight" if dmin > thr else "landed"
            prev_centers = centers
    return tag, id2fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    preds = load_preds(a.preds)
    tag, id2fn = classify_flight(a.ann)
    d = json.load(open(a.ann))
    gt_by_img = collections.defaultdict(list)
    for an in d["annotations"]:
        if an["category_id"] in (LITTER_CID, GARBAGE_CID):
            x, y, w, h = an["bbox"]
            gt_by_img[an["image_id"]].append(
                (an["id"], an["category_id"], [x, y, x+w, y+h]))

    print(f"=== Stage 1 · {a.tag} ===")
    print(f"preds={a.preds.split('/')[-1]}  ann={a.ann.split('/')[-1]}")
    n_f = sum(1 for v in tag.values() if v == "flight")
    print(f"GT 框：飛行 {n_f}、落地 {len(tag)-n_f}、合計 {len(tag)}\n")

    for thr in (0.30, 0.50):
        # cnt[(狀態, 類別)] = [命中, 總數]
        cnt = collections.defaultdict(lambda: [0, 0])
        n_pred_total = 0
        for iid, gts in gt_by_img.items():
            p = preds.get(iid)
            cand = []
            if p is not None:
                for bb, sc, lb in zip(p["bboxes"], p["scores"], p["labels"]):
                    if sc >= thr and (lb+1) in (LITTER_CID, GARBAGE_CID):
                        cand.append([bb, float(sc), int(lb)+1, False])
                n_pred_total += len(cand)
            for aid, cat, gbox in gts:
                st = tag.get((iid, aid), "landed")
                cnt[(st, NAME[cat])][1] += 1
                cnt[(st, "合計")][1] += 1
                best, bi = 0.0, -1
                for i, (bb, sc, c, used) in enumerate(cand):
                    if used or c != cat:
                        continue
                    v = iou(gbox, bb)
                    if v > best:
                        best, bi = v, i
                if best >= 0.5:
                    cand[bi][3] = True
                    cnt[(st, NAME[cat])][0] += 1
                    cnt[(st, "合計")][0] += 1

        print(f"--- score_thr = {thr:.2f}  (預測框總數 {n_pred_total}) ---")
        print(f"{'類別':<10}{'飛行 recall':>22}{'落地 recall':>22}{'倍數':>10}")
        for c in ("litter", "garbage", "合計"):
            hf, nf = cnt[("flight", c)]
            hl, nl = cnt[("landed", c)]
            rf = hf/nf if nf else float("nan")
            rl = hl/nl if nl else float("nan")
            ratio = (rf/rl) if (nl and rl > 0) else float("inf")
            print(f"{c:<10}{hf:>5}/{nf:<5}={rf:>7.3f}   {hl:>5}/{nl:<5}={rl:>7.3f}   {ratio:>8.2f}×")
        print()

if __name__ == "__main__":
    main()
