#!/usr/bin/env python3
"""信心分數門檻 0.1、0.3、0.5 三組 TP/FP/FN，與影片推論使用同一套配對。

兩種「門檻」不要混淆：
  信心分數門檻（score threshold）：分數 ≥ 門檻的預測框才算數。本表比較 0.1、0.3、0.5 三種。
    0.5＝步驟 5～7 的 TP/FP/FN 與第 16、18 次正式數字使用的門檻；0.3＝過去 Co-DETR 影片推論使用的門檻；
    0.1＝接近 YOLO26l 在 valid 上 litter 的最佳 F1 門檻，看 YOLO 自己最有利的情況（實際的最佳門檻從
        <YB>/seed<N>/test_eval/threshold_eval.json 讀出來寫進 md，不寫死）。
  IoU 門檻：預測框與正確答案框重疊（交集÷聯集）≥ 0.5 才算抓對。兩種分數門檻下都固定是 0.5。
AP 不是在某一個門檻下算的：它把推論存下的所有框依分數排序，等於把門檻從最高掃到存下來的最低分，
所以不受「選哪個門檻數 TP/FP/FN」影響，每組只列一次；但它受存檔下限影響（存檔設定是 conf=0.001、每張圖最多 300 個框，
哪一個先擋住就決定實際下限；端到端版的實際下限見 YOLO_baseline_端到端重跑規劃.md 步驟 5 的測量）。

計算全部用 metrics_from_preds.py（AP 用 coco_ap、TP/FP/FN 用 tp_fp_fn），並用 frame_matching 的逐框配對核對加總一致。

用法（容器內，需設 YOLO_HEAD；YOLO 的 pkl 與影片分類表從該版本的資料夾讀）：python metrics_at_thresholds.py --out <dir>
"""
import argparse
import collections
import contextlib
import csv
import io
import json
import os

from pycocotools.coco import COCO

from frame_matching import check_against_reference
from yolo_eval_common import HEAD, HEAD_LABEL, YB, load_metrics_module

H = "/workspace/hdd2_storage"
DT = f"{H}/18th_expt_v18_own_dynamic_bgs_4ch_36e/dual_track_eval"
V17 = "/workspace/datasets/coco/chiayi_v17_own_dynamic_litter_head10_manual/annotations"
V18 = "/workspace/datasets/coco/chiayi_v18_own_dynamic_bgs_122vids/annotations"
ANN = {"train": f"{V17}/train.json", "valid": f"{V17}/valid.json", "test_full": f"{V17}/test.json",
       "dynamic166": f"{V18}/test_dynamic166.json", "static594": f"{V18}/test_static594.json"}
THRS = [0.1, 0.3, 0.5]
TAG = "_".join(str(t) for t in THRS)
FN_DETAIL = [(1, 0.1)]   # 使用者要求列出 FN clip 名稱的 (seed, 門檻)
SHOW = ["litter", "garbage"]


def jobs():
    for s in range(3):
        for sp in ["train", "valid", "test_full"]:
            yield f"YOLO26l {HEAD_LABEL} seed{s}", sp, f"{YB}/seed{s}/full_preds/{sp}.pkl"
        for sp in ["dynamic166", "static594"]:
            yield f"YOLO26l {HEAD_LABEL} seed{s}", sp, f"{YB}/seed{s}/test_eval/{sp}.pkl"
    for sp in ["dynamic166", "static594"]:
        yield "第16次 Co-DETR", sp, f"{DT}/16th_{sp}.pkl"
        yield "第18次 Co-DETR", sp, f"{DT}/18th_{sp}.pkl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    mfp = load_metrics_module()
    gts = {}
    res = []
    for name, sp, pkl in jobs():
        if sp not in gts:
            with contextlib.redirect_stdout(io.StringIO()):
                gts[sp] = COCO(ANN[sp])
        preds = mfp.load_preds(pkl)
        with contextlib.redirect_stdout(io.StringIO()):
            apr = mfp.coco_ap(gts[sp], preds)
        row = {"model": name, "split": sp, "pkl": pkl, "AP50": {c: round(apr[c]["AP@0.5"], 3) for c in SHOW}}
        for t in THRS:
            _, st = check_against_reference(mfp, gts[sp], preds, t)      # 逐框配對與 tp_fp_fn 不一致會直接停止
            row[f"thr{t}"] = {c: st[c] for c in SHOW}
        res.append(row)
        print(f"{name:16s} {sp:10s} " + "  ".join(
            f"{c} AP50={row['AP50'][c]:.3f} " + " ".join(
                f"[{t}] {row[f'thr{t}'][c]['TP']}/{row[f'thr{t}'][c]['FP']}/{row[f'thr{t}'][c]['FN']}" for t in THRS)
            for c in SHOW), flush=True)
    json.dump(res, open(os.path.join(a.out, f"metrics_thr{TAG}.json"), "w"), ensure_ascii=False, indent=2)

    # 說明文字裡的數字即時算出來，不寫死（2026-10-02 前寫死的 0.11/0.12/0.11、0.047 是 NMS 版的值）
    bf = [json.load(open(f"{YB}/seed{s}/test_eval/threshold_eval.json"))["best_f1_thr_from_valid"]["litter"] for s in range(3)]
    ex = next(r for r in res if r["model"].endswith("seed0") and r["split"] == "dynamic166")
    ex_preds = mfp.load_preds(ex["pkl"])
    ex_hi = {k: {f: v[f][v["scores"] >= 0.5] for f in ("bboxes", "scores", "labels")} for k, v in ex_preds.items()}
    with contextlib.redirect_stdout(io.StringIO()):
        ex_ap_hi = mfp.coco_ap(gts["dynamic166"], ex_hi)["litter"]["AP@0.5"]
    ex_n_hi = sum(int(((v["labels"] == 1) & (v["scores"] >= 0.5)).sum()) for v in ex_preds.values())   # labels 0..4，1＝litter

    md = [f"# 信心分數門檻 0.1、0.3、0.5 的 TP / FP / FN（YOLO26l {HEAD_LABEL} 版）\n\n",
          f"> **YOLO 版本**：{'端到端（一對一頭、不做 NMS），正式版' if HEAD == 'e2e' else '一對多頭＋標準 NMS，補充對照'}。\n"
          "> **信心分數門檻**：分數 ≥ 門檻的預測框才算數。0.5 是步驟 5～7 與第 16、18 次正式數字用的門檻；"
          "0.3 是過去 Co-DETR 影片推論用的門檻，比較寬鬆，留下的框較多；"
          f"0.1 接近 YOLO26l 在 valid 上 litter 的最佳 F1 門檻（三個 seed 為 {'、'.join(f'{t:.2f}' for t in bf)}），"
          "用來看 YOLO 自己最有利的情況。第 16、18 次 Co-DETR 的列一併算出 0.1 只是補齊表格，它們的最佳門檻不在這裡。\n"
          "> **IoU 門檻**：預測框與正確答案框重疊 ≥ 0.5 才算抓對；三種分數門檻下都固定 0.5。\n"
          "> **AP@0.5** 不是在某一個信心分數門檻下算的：它把推論時存下的所有框依分數由高到低排序，"
          "等於把門檻從最高一路掃到存下來的最低分，每個門檻都算一次 Precision 與 Recall，再綜合成一個分數。"
          "所以它不受「選 0.1、0.3 還是 0.5 來數 TP/FP/FN」影響，每列只列一次；但它會受存檔下限影響"
          f"（例：YOLO26l seed0 在 dynamic166 分數 ≥ 0.5 的 litter 框有 {ex_n_hi} 個，若只留 0.5 以上，"
          f"litter AP@0.5 會從 {ex['AP50']['litter']:.3f} 變成 {ex_ap_hi:.3f}）。"
          "推論時都用 conf=0.001、每張圖最多 300 個框存檔。\n> 格式為 TP / FP / FN。所有數字由 `metrics_from_preds.py` 計算，"
          "並已核對與影片使用的逐框配對加總一致。\n\n"]
    for sp in ["train", "valid", "test_full", "dynamic166", "static594"]:
        rows = [r for r in res if r["split"] == sp]
        md.append(f"## {sp}\n\n| 模型 | litter AP@0.5 | " + " | ".join(f"litter 門檻 {t}" for t in THRS) + " | garbage AP@0.5 | "
                  + " | ".join(f"garbage 門檻 {t}" for t in THRS) + " |\n|" + "---|" * (2 * len(THRS) + 3) + "\n")
        for r in rows:
            f = lambda c, t: "{TP} / {FP} / {FN}".format(**r[f"thr{t}"][c])
            md.append(f"| {r['model']} | {r['AP50']['litter']:.3f} | " + " | ".join(f('litter', t) for t in THRS) +
                      f" | {r['AP50']['garbage']:.3f} | " + " | ".join(f('garbage', t) for t in THRS) + " |\n")
        md.append("\n")

    # clip 層級分類：讀影片推論產生的 video_classification.csv（render_videos_from_pkl.py 的輸出）
    md.append(f"## clip 層級分類（YOLO26l {HEAD_LABEL} 版影片推論）\n\n"
              "> **clip 分類規則**（只看 litter、garbage）：有 litter/garbage 正確答案框的 clip，任一幀有 litter/garbage 的 TP → **TP**，"
              "一幀都沒有 → **FN**；沒有正確答案框的 clip，任一幀有 litter/garbage 預測框 → **FP**，否則 → **TN**。\n"
              "> 任一幀命中就算，比逐框寬鬆很多，會讓結果看起來比實際好；判斷模型好壞以上面的逐框數字為準。"
              "train 99 部、valid 15 部都有正確答案框，所以只有 TP / FN；test 8 部中有 1 部（`20240818010308_01m05s`）沒有，所以才可能出現 FP / TN。\n"
              "> 來源：`seed<N>/video_inference/thr<門檻>/video_classification.csv`。\n\n"
              "| seed | 信心分數門檻 | train（99 部）TP / FN | valid（15 部）TP / FN | test（8 部）TP / FP / FN / TN |\n|---|---|---|---|---|\n")
    for s in range(3):
        for t in THRS:
            with open(f"{YB}/seed{s}/video_inference/thr{t}/video_classification.csv", encoding="utf-8") as fh:
                cr = list(csv.DictReader(fh))
            n = {sp: collections.Counter(r["classification"] for r in cr if r["split"] == sp) for sp in ["train", "valid", "test"]}
            md.append(f"| {s} | {t} | {n['train']['TP']} / {n['train']['FN']} | {n['valid']['TP']} / {n['valid']['FN']} | "
                      f"{n['test']['TP']} / {n['test']['FP']} / {n['test']['FN']} / {n['test']['TN']} |\n")
    md.append("\n")
    for s, t in FN_DETAIL:
        with open(f"{YB}/seed{s}/video_inference/thr{t}/video_classification.csv", encoding="utf-8") as fh:
            fn = [r for r in csv.DictReader(fh) if r["classification"] == "FN" and r["split"] in ("train", "valid")]
        md.append(f"### seed {s}、門檻 {t}：train 與 valid 的 FN clip（共 {len(fn)} 部）\n\n"
                  "> 這些 clip 有 litter/garbage 正確答案框，但每一幀的 litter/garbage TP 都是 0。"
                  "「漏掉的框」＝該 clip 所有幀加總的 FN 框數；「誤報的框」＝分數 ≥ 門檻、但沒配到正確答案的預測框數（FP）。\n\n"
                  "| split | clip | 幀數 | litter 漏掉的框 | litter 誤報的框 | garbage 漏掉的框 | garbage 誤報的框 |\n|---|---|---|---|---|---|---|\n")
        for r in fn:
            md.append(f"| {r['split']} | `{r['video_name']}` | {r['num_frames']} | {r['litter_FN']} | {r['litter_FP']} | "
                      f"{r['garbage_FN']} | {r['garbage_FP']} |\n")
        md.append("\n")
    open(os.path.join(a.out, f"metrics_thr{TAG}.md"), "w", encoding="utf-8").writelines(md)
    print("METRICS_THR_DONE →", a.out)


if __name__ == "__main__":
    main()
