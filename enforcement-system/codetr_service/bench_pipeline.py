#!/usr/bin/env python3
"""把 changemap 流程五個步驟的每幀耗時，用同一次掃描、同一支影片、同一批幀量出來。

先前 優化_rm5向量化.md 第 2 節那張表的五個數字來自三次不同的量測
（步驟 1-3 來自 20 幀的快速版、步驟 4 來自全片版、步驟 5 用的還是合成遮罩），
拼在一起會讓人以為是同一次跑出來的。這支腳本把五個步驟一次量完。

量測對象：537_8_loc537_20240707063733.avi
          （廠商交付的 113.7.7 批次，案57；同名的「案57」在 6/8/9 月批次各有一支
            不同影片，所以引用時一律用檔名）

步驟 1-3 對「掃過的每一幀」逐幀計時（暖機後 14,884 幀），取全部的中位數。
步驟 4-5 對取樣幀（每 150 幀一張，共 100 張）各跑 3 次取最快，再取中位數
—— 因為這兩步是純運算、可以重複，取最快能避開排程雜訊；
   步驟 1-3 含 I/O 與背景模型狀態，不能重跑同一幀，只能逐幀記錄。
"""
import time, json, statistics as st
import cv2, numpy as np

AVI = ("/path/to/data/raw_videos/7月份影片/113.7.7 OK/"
       "case57/"
       "537_8_loc537_20240707063733.avi")
S, HIST, DIST, RM, WARM, SAMPLE_EVERY = 0.5, 500, 500.0, 5, 150, 150


def rm5(mask):
    fg = (mask == 255).astype(np.uint8)
    n, lab, stt, _ = cv2.connectedComponentsWithStats(fg, 8)
    keep = np.zeros(n, dtype=np.uint8)
    keep[1:] = (stt[1:, cv2.CC_STAT_AREA] >= RM)
    return (keep[lab] * 255).astype(np.uint8)


def fastest(fn, arg, k=3):
    best = float("inf")
    for _ in range(k):
        t = time.perf_counter(); fn(arg); best = min(best, time.perf_counter() - t)
    return best * 1e3


cap = cv2.VideoCapture(AVI)
fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ok, f0 = cap.read()
print(f"影片 {AVI.rsplit('/',1)[1]}", flush=True)
print(f"  {total} 幀 / {fps} fps = {total/fps/60:.1f} 分，原始 {f0.shape[1]}x{f0.shape[0]}", flush=True)
cap.release()

cap = cv2.VideoCapture(AVI)
knn = cv2.createBackgroundSubtractorKNN(history=HIST, dist2Threshold=DIST, detectShadows=True)
for _ in range(WARM):
    ok, f = cap.read()
    knn.apply(cv2.resize(f, None, fx=S, fy=S))
print(f"暖機 {WARM} 幀完成，開始逐幀計時", flush=True)

t_dec, t_down, t_knn, t_rm5, t_up = [], [], [], [], []
idx, t0 = WARM, time.perf_counter()
while True:
    t = time.perf_counter(); ok, f = cap.read(); dt = (time.perf_counter() - t) * 1e3
    if not ok:
        break
    t_dec.append(dt)
    t = time.perf_counter(); small = cv2.resize(f, None, fx=S, fy=S); t_down.append((time.perf_counter()-t)*1e3)
    t = time.perf_counter(); m = knn.apply(small, learningRate=0);   t_knn.append((time.perf_counter()-t)*1e3)
    if (idx - WARM) % SAMPLE_EVERY == 0:
        H, W = f.shape[:2]
        t_rm5.append(fastest(rm5, m))
        cm = rm5(m)
        t_up.append(fastest(lambda x: cv2.resize(x, (W, H), interpolation=cv2.INTER_NEAREST), cm))
    idx += 1
cap.release()
n_scan = idx - WARM
print(f"掃完 {n_scan} 幀（{time.perf_counter()-t0:.0f}s），步驟 4-5 取樣 {len(t_rm5)} 幀\n", flush=True)


def row(name, v, how):
    v = sorted(v)
    p = lambda q: v[min(len(v)-1, int(q*len(v)))]
    return dict(step=name, n=len(v), how=how, min=p(0), p25=p(.25), med=p(.5),
                p75=p(.75), p95=p(.95), max=p(.999))


rows = [
    row("1 解碼",        t_dec,  f"逐幀計時，全部 {n_scan} 幀"),
    row("2 縮成半解析度", t_down, f"逐幀計時，全部 {n_scan} 幀"),
    row("3 KNN 背景相減", t_knn,  f"逐幀計時，全部 {n_scan} 幀"),
    row("4 rm5（查表版）", t_rm5, f"取樣 {len(t_rm5)} 幀，各 3 次取最快"),
    row("5 放大回原尺寸",  t_up,  f"取樣 {len(t_up)} 幀，各 3 次取最快"),
]
print(f"{'步驟':16}{'樣本數':>8}{'最小':>9}{'p25':>9}{'中位數':>9}{'p75':>9}{'p95':>9}{'最大':>10}  怎麼量")
for r in rows:
    print(f"{r['step']:16}{r['n']:>8}" + "".join(f"{r[k]:>9.2f}" for k in
          ("min", "p25", "med", "p75", "p95")) + f"{r['max']:>10.2f}  {r['how']}")

tot_med = sum(r["med"] for r in rows)
print(f"\n五步中位數相加 = {tot_med:.2f} 毫秒/幀（不含 Co-DETR 推論）")
print(f"其中 rm5 佔 {rows[3]['med']/tot_med*100:.1f}%")

json.dump(dict(video=AVI, frames_scanned=n_scan, rows=rows),
          open("bench_pipeline.json", "w"), ensure_ascii=False, indent=1)
print("\n明細寫入 bench_pipeline.json")
