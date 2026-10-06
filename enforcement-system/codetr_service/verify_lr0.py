#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在真實影片上驗證：apply(learningRate=0) 到底是不是「凍結」。

量測語意刻意與線上一致：每次 apply 同時是「更新」也是「量測」，
不做額外的 query apply —— 因為 query 本身也會更新模型，
分開做會讓有效更新率變成 1.5 倍（2026-08-23 第一版腳本就是這樣量壞的）。

輸出「吸收所需的 apply 次數」，而不是秒數：
若吸收只跟 apply 次數有關、與影片時間無關，那 apply 間隔拉大 35 倍，
吸收所需的次數應該不變 —— 這正是要驗的事。
"""
import cv2, numpy as np, json, sys

V = ("/path/to/data/raw_videos/7月份影片/113.7.7 OK/"
     "case57/"
     "537_8_loc537_20240707063733.avi")
S, HIST, DIST, WARM = 0.5, 500, 500.0, 150
SITE = (1884, 1072)      # 中段路面（verify 過落在柏油上）
PATCH = 40               # 色塊邊長（全解析度 px）
COLOR = (20, 20, 20)
START = 3000             # 第 300 秒開始貼色塊
ALPHAS = [0.0, 0.0002, 0.001, 0.002, 0.02]
STRIDES = [1, 35]        # 1 = 每幀 apply；35 = 線上 pipeline 的實際取樣間隔

cap = cv2.VideoCapture(V)
if not cap.isOpened(): sys.exit("開不了影片")
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

runs = {}   # (stride, alpha) -> dict
for st in STRIDES:
    for a in ALPHAS:
        runs[(st, a)] = {"knn": cv2.createBackgroundSubtractorKNN(
            history=HIST, dist2Threshold=DIST, detectShadows=True),
            "n_apply": 0, "hit": None, "first": None, "series": []}

ps = int(PATCH * S); py, px = int(SITE[1] * S), int(SITE[0] * S)
print(f"影片 {N} 幀；色塊 {PATCH}px @ {SITE}；第 {START} 幀開始貼", flush=True)

for idx in range(N):
    ok, frame = cap.read()
    if not ok: break

    if idx < WARM:
        small = cv2.resize(frame, None, fx=S, fy=S)
        for r in runs.values(): r["knn"].apply(small)
        continue

    if idx >= START:
        frame[SITE[1]:SITE[1]+PATCH, SITE[0]:SITE[0]+PATCH] = COLOR
    small = cv2.resize(frame, None, fx=S, fy=S)

    for (st, a), r in runs.items():
        if (idx - WARM) % st: continue
        m = r["knn"].apply(small, learningRate=a)   # 更新＋量測合一，同線上語意
        if idx < START: continue
        r["n_apply"] += 1
        fg = float((m[py:py+ps, px:px+ps] == 255).mean())
        if r["first"] is None: r["first"] = fg
        if len(r["series"]) < 40: r["series"].append(round(fg, 3))
        if r["hit"] is None and r["first"] > 0.5 and fg < 0.5:
            r["hit"] = r["n_apply"]

    if idx % 3000 == 0 and idx: print(f"  {idx}/{N}", flush=True)

cap.release()

print("\n" + "="*76)
print("色塊貼上後，需要幾次 apply 才被吃進背景（前景比例掉到 50% 以下）")
print("="*76)
print(f"{'α':>9} | {'每幀 apply':>22} | {'每 35 幀 apply（線上）':>26}")
print("-"*76)
for a in ALPHAS:
    cells = []
    for st in STRIDES:
        r = runs[(st, a)]
        cells.append(f"{r['hit']} 次" if r["hit"] else f">{r['n_apply']} 次(未吸收)")
    print(f"{a:>9} | {cells[0]:>22} | {cells[1]:>26}")

print("\n前 12 次 apply 的色塊前景比例（每幀 apply）：")
for a in ALPHAS:
    print(f"  α={a:<8} " + " ".join(f"{v:.2f}" for v in runs[(1,a)]["series"][:12]))

json.dump({f"{st}_{a}": {"hit": runs[(st,a)]["hit"],
                         "n_apply": runs[(st,a)]["n_apply"],
                         "series": runs[(st,a)]["series"]}
           for st in STRIDES for a in ALPHAS},
          open("verify_lr0.json","w"), ensure_ascii=False, indent=1)
print("\n寫入 verify_lr0.json")
