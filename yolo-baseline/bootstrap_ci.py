#!/usr/bin/env python3
"""步驟 7：配對 bootstrap 信賴區間（YOLO_baseline_定案規劃.md 的 D5）。

比較兩組差值的 AP@0.5（litter、garbage 各自）：
  (1) YOLO26l（3 個 seed 的平均）− 第 16 次 Co-DETR       ← 比較 1：架構
  (2) 第 18 次 Co-DETR − 第 16 次 Co-DETR                 ← 比較 2：BGS
每一輪重抽一批 test，所有模型在「同一批」上各算一次 AP，記下差值；重複 N 次，取第 2.5 與 97.5 百分位數。

重抽單位兩種：
  clip（主要）：一次抽整部 clip 的所有幀。test 的幀來自少數 clip、相鄰幀高度相關，以 clip 為單位才不會高估可靠度。
  frame（參考）：以單張幀為單位，會把相關的幀當成獨立樣本，區間偏窄。
重抽後如果整批裡完全沒有該類別的正確答案框，AP 無定義，這一輪不計入，並記錄被略過的輪數。

AP 的算法：先用 pycocotools 的 COCOeval.evaluate() 算出每張圖的配對結果（這一步本來就逐張獨立），
每輪只依重抽結果重組，再照 pycocotools COCOeval.accumulate() 的原始碼算 AP@0.5（area=all、maxDet=100、
101 點插值）。腳本開頭會先驗證「不重抽時算出的 AP 與 pycocotools / metrics_from_preds 完全相同」，不同就停止。

用法（容器內，需設 YOLO_HEAD，YOLO 的 pkl 從該版本的資料夾讀）：python bootstrap_ci.py --out <dir> [--n 1000]
"""
import argparse
import contextlib
import io
import json
import os
import re

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from yolo_eval_common import YB, load_metrics_module

H = "/workspace/hdd2_storage"
V18 = "/workspace/datasets/coco/chiayi_v18_own_dynamic_bgs_122vids/annotations"
DT = f"{H}/18th_expt_v18_own_dynamic_bgs_4ch_36e/dual_track_eval"
SPLITS = {"dynamic166": f"{V18}/test_dynamic166.json", "static594": f"{V18}/test_static594.json"}
MODELS = {
    "codetr16": lambda sp: f"{DT}/16th_{sp}.pkl",
    "codetr18": lambda sp: f"{DT}/18th_{sp}.pkl",
    "yolo_s0": lambda sp: f"{YB}/seed0/test_eval/{sp}.pkl",
    "yolo_s1": lambda sp: f"{YB}/seed1/test_eval/{sp}.pkl",
    "yolo_s2": lambda sp: f"{YB}/seed2/test_eval/{sp}.pkl",
}
CLASSES = {"litter": 2, "garbage": 5}
REC_THRS = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)   # 與 pycocotools 相同


def per_image_eval(coco_gt, preds, cat_id):
    """回傳 {img_id: evalImg dict 或 None}（area=all）。"""
    dets = []
    for img_id, p in preds.items():
        for bb, sc, lb in zip(p["bboxes"], p["scores"], p["labels"]):
            if int(lb) + 1 != cat_id:
                continue
            x1, y1, x2, y2 = bb
            dets.append(dict(image_id=int(img_id), category_id=cat_id,
                             bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)], score=float(sc)))
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(dets) if dets else COCO()
        E = COCOeval(coco_gt, coco_dt, "bbox")
        E.params.catIds = [cat_id]
        E.evaluate()
    I = len(E.params.imgIds)
    return {img: E.evalImgs[i] for i, img in enumerate(E.params.imgIds)}  # 索引 k=0、a=0(all) 的那段就是前 I 個


def ap50_from(evals, img_list, max_det=100):
    """照 pycocotools accumulate() 對單一類別、IoU=0.5、area=all 算 AP；無正確答案框時回傳 None。"""
    E = [evals[i] for i in img_list if evals[i] is not None]
    if not E:
        return None
    dt_scores = np.concatenate([e["dtScores"][0:max_det] for e in E])
    inds = np.argsort(-dt_scores, kind="mergesort")
    dtm = np.concatenate([e["dtMatches"][:, 0:max_det] for e in E], axis=1)[:, inds]
    dt_ig = np.concatenate([e["dtIgnore"][:, 0:max_det] for e in E], axis=1)[:, inds]
    gt_ig = np.concatenate([e["gtIgnore"] for e in E])
    npig = np.count_nonzero(gt_ig == 0)
    if npig == 0:
        return None
    t = 0                                                                   # IoU 0.5
    tps = np.logical_and(dtm[t], np.logical_not(dt_ig[t]))
    fps = np.logical_and(np.logical_not(dtm[t]), np.logical_not(dt_ig[t]))
    tp = np.cumsum(tps).astype(float)
    fp = np.cumsum(fps).astype(float)
    nd = len(tp)
    rc = tp / npig
    pr = (tp / (fp + tp + np.spacing(1))).tolist()
    q = np.zeros(len(REC_THRS))
    for i in range(nd - 1, 0, -1):
        if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]
    idx = np.searchsorted(rc, REC_THRS, side="left")
    for ri, pi in enumerate(idx):
        if pi < nd:
            q[ri] = pr[pi]
    return float(np.mean(q))


def clip_of(file_name):
    return re.sub(r"_frame_\d+\.jpg$", "", file_name)


def ci(diffs):
    d = np.sort(np.asarray(diffs))
    return float(d[int(np.floor(0.025 * len(d)))]), float(d[int(np.ceil(0.975 * len(d))) - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(2026)
    mfp = load_metrics_module()
    report = {"n_resamples": a.n, "metric": "AP@0.5", "results": {}}

    for sp, ann in SPLITS.items():
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(ann)
        imgs = sorted(gt.getImgIds())
        clips = {}
        for i in imgs:
            clips.setdefault(clip_of(gt.imgs[i]["file_name"]), []).append(i)
        clip_names = sorted(clips)
        preds = {m: mfp.load_preds(f(sp)) for m, f in MODELS.items()}

        for cname, cid in CLASSES.items():
            evals = {m: per_image_eval(gt, preds[m], cid) for m in MODELS}

            # 驗證：完整 test 集上，本腳本的 AP 必須等於 metrics_from_preds（pycocotools）的 AP@0.5
            with contextlib.redirect_stdout(io.StringIO()):
                ref = {m: mfp.coco_ap(gt, preds[m])[cname]["AP@0.5"] for m in MODELS}
            full = {m: ap50_from(evals[m], imgs) for m in MODELS}
            for m in MODELS:
                assert abs(full[m] - ref[m]) < 1e-9, f"{sp}/{cname}/{m}: 自算 AP {full[m]} ≠ pycocotools {ref[m]}"
            print(f"[驗證通過] {sp} {cname}：自算 AP 與 pycocotools 完全一致 "
                  + "，".join(f"{m}={full[m]:.4f}" for m in MODELS), flush=True)

            yolo_mean = lambda d: np.mean([d["yolo_s0"], d["yolo_s1"], d["yolo_s2"]])
            point = {"yolo_mean_minus_16": yolo_mean(full) - full["codetr16"],
                     "18_minus_16": full["codetr18"] - full["codetr16"]}
            res = {"full_test_AP50": full, "point_diff": point}

            for level in ["clip", "frame"]:
                diffs = {"yolo_mean_minus_16": [], "18_minus_16": []}
                skipped = 0
                for _ in range(a.n):
                    if level == "clip":
                        pick = rng.choice(len(clip_names), size=len(clip_names), replace=True)
                        sample = [i for k in pick for i in clips[clip_names[k]]]
                    else:
                        sample = list(rng.choice(imgs, size=len(imgs), replace=True))
                    aps = {m: ap50_from(evals[m], sample) for m in MODELS}
                    if any(v is None for v in aps.values()):
                        skipped += 1
                        continue
                    diffs["yolo_mean_minus_16"].append(yolo_mean(aps) - aps["codetr16"])
                    diffs["18_minus_16"].append(aps["codetr18"] - aps["codetr16"])
                res[level] = {k: {"ci95": ci(v), "n_valid": len(v)} for k, v in diffs.items()}
                res[level]["skipped_no_gt"] = skipped
                for k, v in diffs.items():
                    lo, hi = ci(v)
                    verdict = "跨過 0（無法確定）" if lo <= 0 <= hi else ("整個 > 0" if lo > 0 else "整個 < 0")
                    print(f"  {sp:10s} {cname:7s} {level:5s} {k:20s} 點估計 {point[k]:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
                          f"{verdict}  (有效 {len(v)} 輪，略過 {skipped})", flush=True)
            res["n_clips"] = len(clip_names)
            res["clips_with_class"] = sorted({clip_of(gt.imgs[g["image_id"]]["file_name"])
                                              for g in gt.dataset["annotations"] if g["category_id"] == cid})
            report["results"][f"{sp}/{cname}"] = res

    json.dump(report, open(os.path.join(a.out, "bootstrap_ci.json"), "w"), ensure_ascii=False, indent=2)
    print("BOOTSTRAP_DONE →", a.out)


if __name__ == "__main__":
    main()
