#!/usr/bin/env python3
"""端到端 vs 一對多頭＋NMS 的對照表（見 ../YOLO_baseline_端到端重跑規劃.md 步驟 5b 與 3-4）。

只讀已經算好的 JSON，不推論，在主機上直接跑（python3 head_compare.py）。分成三張表，彼此不要混在一起：
  (a) 同一個 checkpoint 只換頭（test）：端到端版選中的 epoch 上，e2e 頭 vs nms 頭。回答「用不用 NMS 有沒有差」。
      nms 頭的結果：若兩版選中同一個 epoch，用 nms/seed<N>/test_eval；否則用步驟 5b 補跑的 e2e/head_compare/seed<N>/nms_ep<B>/
  (a') 同一個 checkpoint 只換頭（valid，100 個 epoch 全部）：兩版 epoch_select/valid_ap_per_epoch.json 逐 epoch 相減。
  (b) 兩版各自挑 epoch 後的正式結果（test）：回答「最後報出來的數字差多少」，差異混有「頭不同」與「epoch 不同」兩個來源。
輸出：e2e/head_compare/head_compare.{json,md}
"""
import json
import os

import numpy as np

ROOT = "/path/to/data/baseline_yolo26l_v17"
OUT = f"{ROOT}/e2e/head_compare"
CLASSES = ["car", "litter", "motorcycle", "person", "garbage"]
SPLITS = ["dynamic166", "static594"]


def best(head, s):
    return json.load(open(f"{ROOT}/{head}/seed{s}/epoch_select/best_epoch.json"))["best_epoch"]


def rows(path):
    return json.load(open(path))["rows"]


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {"a_same_ckpt_test": {}, "a_same_ckpt_valid": {}, "b_official_test": {}}
    md = ["# YOLO26l 端到端 vs 一對多頭＋NMS\n\n",
          "> 端到端（e2e）＝一對一頭、不做 NMS；NMS＝一對多頭＋標準 NMS（IoU 0.7）。兩個頭在同一次訓練裡一起訓練，"
          "權重檔相同，推論時只差用哪個頭。AP@0.5 由 metrics_from_preds.py（pycocotools）計算；TP/FP/FN 是信心分數門檻 0.5、IoU 0.5。\n\n"]

    # (a) 同 checkpoint，test
    md.append("## (a) 同一個 checkpoint 只換頭（test）\n\n"
              "> checkpoint＝端到端版在 valid 上選中的 epoch。差值＝e2e − NMS。\n\n"
              "| seed | epoch | 測試集 | 類別 | NMS AP@0.5 | e2e AP@0.5 | 差值 | NMS TP/FP/FN | e2e TP/FP/FN |\n|---|---|---|---|---|---|---|---|---|\n")
    for s in range(3):
        be, bn = best("e2e", s), best("nms", s)
        nms_dir = f"{ROOT}/nms/seed{s}/test_eval" if be == bn else f"{OUT}/seed{s}/nms_ep{be}"
        for sp in SPLITS:
            e = rows(f"{ROOT}/e2e/seed{s}/test_eval/{sp}_metrics.json")
            n = rows(f"{nms_dir}/{sp}_metrics.json")
            for c in CLASSES:
                res["a_same_ckpt_test"].setdefault(f"seed{s}", {"epoch": be, "nms_source": nms_dir})[f"{sp}/{c}"] = \
                    {"nms": n[c], "e2e": e[c], "diff_AP50": round(e[c]["AP50"] - n[c]["AP50"], 3)}
                tf = (lambda r: f"{r['TP']}/{r['FP']}/{r['FN']}") if c in ("litter", "garbage") else (lambda r: "—")
                md.append(f"| {s} | {be} | {sp} | {c} | {n[c]['AP50']:.3f} | {e[c]['AP50']:.3f} | {e[c]['AP50'] - n[c]['AP50']:+.3f} | {tf(n[c])} | {tf(e[c])} |\n")
    # 三個 seed 平均
    md.append("\n三個 seed 平均（同一個 checkpoint 只換頭）：\n\n| 測試集 | 類別 | NMS AP@0.5 平均 | e2e AP@0.5 平均 | 差值 |\n|---|---|---|---|---|\n")
    for sp in SPLITS:
        for c in CLASSES:
            v = [res["a_same_ckpt_test"][f"seed{s}"][f"{sp}/{c}"] for s in range(3)]
            mn, me = np.mean([x["nms"]["AP50"] for x in v]), np.mean([x["e2e"]["AP50"] for x in v])
            md.append(f"| {sp} | {c} | {mn:.3f} | {me:.3f} | {me - mn:+.3f} |\n")

    # (a') 同 checkpoint，valid 每個 epoch
    md.append("\n## (a') 同一個 checkpoint 只換頭（valid，100 個 epoch 逐一比較）\n\n"
              "> 每個 epoch 兩個頭各算一次 valid AP，相減（e2e − NMS）後統計 100 個 epoch。\n\n"
              "| seed | 指標 | NMS 平均 | e2e 平均 | 差值平均 | 差值範圍 | e2e 較高的 epoch 數 |\n|---|---|---|---|---|---|---|\n")
    for s in range(3):
        e = json.load(open(f"{ROOT}/e2e/seed{s}/epoch_select/valid_ap_per_epoch.json"))
        n = json.load(open(f"{ROOT}/nms/seed{s}/epoch_select/valid_ap_per_epoch.json"))
        assert sorted(e, key=int) == sorted(n, key=int) and len(e) == 100
        for c, m in [("litter", "AP@0.5:0.95"), ("litter", "AP@0.5"), ("garbage", "AP@0.5"), ("car", "AP@0.5")]:
            ev = np.array([e[k][c][m] for k in sorted(e, key=int)])
            nv = np.array([n[k][c][m] for k in sorted(n, key=int)])
            d = ev - nv
            res["a_same_ckpt_valid"][f"seed{s}/{c}/{m}"] = dict(nms_mean=float(nv.mean()), e2e_mean=float(ev.mean()),
                                                                 diff_mean=float(d.mean()), diff_min=float(d.min()),
                                                                 diff_max=float(d.max()), n_e2e_higher=int((d > 0).sum()))
            md.append(f"| {s} | {c} {m} | {nv.mean():.4f} | {ev.mean():.4f} | {d.mean():+.4f} | {d.min():+.4f} ～ {d.max():+.4f} | {(d > 0).sum()} / 100 |\n")

    # (b) 各自挑 epoch 的正式結果
    md.append("\n## (b) 兩版各自挑 epoch 後的正式結果（test）\n\n"
              "> 差異同時來自「頭不同」與「選中的 epoch 不同」。\n\n"
              "| seed | NMS epoch | e2e epoch | 測試集 | 類別 | NMS AP@0.5 | e2e AP@0.5 | 差值 | NMS TP/FP/FN | e2e TP/FP/FN |\n|---|---|---|---|---|---|---|---|---|---|\n")
    for s in range(3):
        be, bn = best("e2e", s), best("nms", s)
        for sp in SPLITS:
            e = rows(f"{ROOT}/e2e/seed{s}/test_eval/{sp}_metrics.json")
            n = rows(f"{ROOT}/nms/seed{s}/test_eval/{sp}_metrics.json")
            for c in ["litter", "garbage", "car"]:
                res["b_official_test"].setdefault(f"seed{s}", {"epoch_nms": bn, "epoch_e2e": be})[f"{sp}/{c}"] = \
                    {"nms": n[c], "e2e": e[c], "diff_AP50": round(e[c]["AP50"] - n[c]["AP50"], 3)}
                tf = (lambda r: f"{r['TP']}/{r['FP']}/{r['FN']}") if c in ("litter", "garbage") else (lambda r: "—")
                md.append(f"| {s} | {bn} | {be} | {sp} | {c} | {n[c]['AP50']:.3f} | {e[c]['AP50']:.3f} | {e[c]['AP50'] - n[c]['AP50']:+.3f} | {tf(n[c])} | {tf(e[c])} |\n")

    json.dump(res, open(f"{OUT}/head_compare.json", "w"), ensure_ascii=False, indent=2)
    open(f"{OUT}/head_compare.md", "w", encoding="utf-8").writelines(md)
    print("".join(md))
    print("HEAD_COMPARE_DONE")


if __name__ == "__main__":
    main()
