#!/usr/bin/env python3
"""步驟 7：逐 clip 結果表（YOLO_baseline_定案規劃.md 的 D6）。

test 集的 litter 只來自 3 部 clip、garbage 也只來自 3 部。這張表把每個模型在每部 clip 上的 TP / FN 攤開，
讓讀者看出模型之間的差距是平均分布，還是集中在某一部 clip。

TP/FN 用 metrics_from_preds.py 自己的 tp_fp_fn()（只把 test 標註裁成該 clip 的影像），報兩種門檻：
  門檻 0.5：與正式數字同一把尺
  best-F1 門檻：步驟 6 在 valid 上決定的（第 16 次與 YOLO 各 seed 有；第 18 次沒有 valid 預測，只報 0.5）

用法（容器內，需設 YOLO_HEAD）：python per_clip_table.py --out <dir>
"""
import argparse
import contextlib
import copy
import io
import json
import os
import re

from pycocotools.coco import COCO

from yolo_eval_common import HEAD_LABEL, YB, YB_SHARED, load_metrics_module

H = "/workspace/hdd2_storage"
V18 = "/workspace/datasets/coco/chiayi_v18_own_dynamic_bgs_122vids/annotations"
DT = f"{H}/18th_expt_v18_own_dynamic_bgs_4ch_36e/dual_track_eval"
SPLITS = {"dynamic166": f"{V18}/test_dynamic166.json", "static594": f"{V18}/test_static594.json"}
MODELS = [  # (顯示名稱, pkl 路徑樣板, threshold_eval.json 或 None)
    ("第16次 Co-DETR", f"{DT}/16th_{{sp}}.pkl", f"{YB_SHARED}/codetr16_valid/threshold_eval.json"),
    ("第18次 Co-DETR", f"{DT}/18th_{{sp}}.pkl", None),
    (f"YOLO26l {HEAD_LABEL} seed0", f"{YB}/seed0/test_eval/{{sp}}.pkl", f"{YB}/seed0/test_eval/threshold_eval.json"),
    (f"YOLO26l {HEAD_LABEL} seed1", f"{YB}/seed1/test_eval/{{sp}}.pkl", f"{YB}/seed1/test_eval/threshold_eval.json"),
    (f"YOLO26l {HEAD_LABEL} seed2", f"{YB}/seed2/test_eval/{{sp}}.pkl", f"{YB}/seed2/test_eval/threshold_eval.json"),
]
CLASSES = {"litter": 2, "garbage": 5}


def clip_of(fn):
    return re.sub(r"_frame_\d+\.jpg$", "", fn)


def short(clip):
    m = re.search(r"(\d{14}_\d+m\d+s)", clip)
    return m.group(1) if m else clip


def sub_coco(full, img_ids):
    d = copy.deepcopy(full.dataset)
    keep = set(img_ids)
    d["images"] = [im for im in d["images"] if im["id"] in keep]
    d["annotations"] = [a for a in d["annotations"] if a["image_id"] in keep]
    c = COCO()
    c.dataset = d
    with contextlib.redirect_stdout(io.StringIO()):
        c.createIndex()
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mfp = load_metrics_module()
    out = {}
    md = []
    for sp, ann in SPLITS.items():
        with contextlib.redirect_stdout(io.StringIO()):
            gt = COCO(ann)
        clips = {}
        for i in gt.getImgIds():
            clips.setdefault(clip_of(gt.imgs[i]["file_name"]), []).append(i)
        preds = {n: mfp.load_preds(p.format(sp=sp)) for n, p, _ in MODELS}
        thr = {n: (json.load(open(t))["best_f1_thr_from_valid"] if t else None) for n, _, t in MODELS}
        for cname, cid in CLASSES.items():
            rel = [c for c in sorted(clips) if any(g["category_id"] == cid for i in clips[c] for g in gt.imgToAnns[i])]
            key = f"{sp}/{cname}"
            out[key] = {}
            md.append(f"\n### {sp}・{cname}（格內為 TP / FN；括號內為 best-F1 門檻下的 TP / FN）\n")
            md.append("| clip | 幀數 | 正確答案框 | " + " | ".join(n for n, _, _ in MODELS) + " |")
            md.append("|---|---|---|" + "---|" * len(MODELS))
            for c in rel:
                sg = sub_coco(gt, clips[c])
                n_gt = sum(1 for g in sg.dataset["annotations"] if g["category_id"] == cid)
                cells = []
                out[key][c] = {"frames": len(clips[c]), "gt": n_gt}
                for n, _, _ in MODELS:
                    mfp.SCORE_THR = 0.5
                    s05 = mfp.tp_fp_fn(sg, preds[n])[cname]
                    cell = f"{s05['TP']} / {s05['FN']}"
                    rec = {"thr0.5": s05}
                    if thr[n]:
                        mfp.SCORE_THR = float(thr[n][cname])
                        sb = mfp.tp_fp_fn(sg, preds[n])[cname]
                        cell += f"（{sb['TP']} / {sb['FN']}）"
                        rec["best_f1"] = dict(thr=float(thr[n][cname]), **sb)
                    mfp.SCORE_THR = 0.5
                    out[key][c][n] = rec
                    cells.append(cell)
                md.append(f"| {short(c)} | {len(clips[c])} | {n_gt} | " + " | ".join(cells) + " |")
    json.dump(out, open(os.path.join(a.out, "per_clip_table.json"), "w"), ensure_ascii=False, indent=2)
    open(os.path.join(a.out, "per_clip_table.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print("\n".join(md))
    print("PER_CLIP_DONE")


if __name__ == "__main__":
    main()
