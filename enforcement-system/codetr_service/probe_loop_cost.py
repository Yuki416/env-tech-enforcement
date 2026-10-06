#!/usr/bin/env python3
"""查清楚迴圈版的耗時到底由什麼決定：元件總數，還是「通過門檻被保留」的元件數？

起因：優化_rm5向量化.md 第 4 節原本寫「成本 ≈ 元件數 × 全圖掃描一次」，
但拿實測的每格成本回推，4,139 個元件應該要 1,717 毫秒，實測中位數卻只有 125 毫秒，
差了一個數量級 —— 表示那個機制描述是錯的。

真正的關鍵是迴圈裡的 if：
    for i in range(1, n):
        if st[i, CC_STAT_AREA] >= RM:     # 大多數元件在這裡就被刷掉
            out[lab == i] = 1             # 只有通過的才付「掃一次全圖」

所以要分開量三件事：
    t_loop   完整迴圈版
    t_check  只跑 Python 迴圈與面積比較，不做任何全圖操作
    t_vec    查表版
再看 t_loop 跟「元件總數」與「保留數」各自的相關係數哪個強。

素材：537_8_loc537_20240707063733.avi 全片，每 150 幀取一張遮罩。
"""
import time, json, statistics as st
import cv2, numpy as np

AVI = ("/path/to/data/raw_videos/7月份影片/113.7.7 OK/"
       "case57/"
       "537_8_loc537_20240707063733.avi")
S, HIST, DIST, RM, WARM, EVERY, REPEAT = 0.5, 500, 500.0, 5, 150, 150, 3


def fastest(fn, k=REPEAT):
    b = float("inf")
    for _ in range(k):
        t = time.perf_counter(); fn(); b = min(b, time.perf_counter() - t)
    return b * 1e3


def probe(m):
    fg = (m == 255).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    area = stats[:, cv2.CC_STAT_AREA]

    def loop():
        out = np.zeros_like(fg)
        for i in range(1, n):
            if area[i] >= RM:
                out[lab == i] = 1
        return (out * 255).astype(np.uint8)

    def check():                      # 只有 Python 迴圈 + 面積比較
        c = 0
        for i in range(1, n):
            if area[i] >= RM:
                c += 1
        return c

    def vec():
        keep = np.zeros(n, dtype=np.uint8)
        keep[1:] = (area[1:] >= RM)
        return (keep[lab] * 255).astype(np.uint8)

    return dict(n_comp=n - 1,
                n_kept=int((area[1:] >= RM).sum()) if n > 1 else 0,
                fg_pct=float((m == 255).sum()) / m.size * 100,
                t_loop=fastest(loop), t_check=fastest(check), t_vec=fastest(vec))


print(f"cv2 {cv2.__version__} / numpy {np.__version__}", flush=True)
cap = cv2.VideoCapture(AVI)
knn = cv2.createBackgroundSubtractorKNN(history=HIST, dist2Threshold=DIST, detectShadows=True)
for _ in range(WARM):
    ok, f = cap.read(); knn.apply(cv2.resize(f, None, fx=S, fy=S))
masks, i = [], 0
while True:
    ok, f = cap.read()
    if not ok:
        break
    m = knn.apply(cv2.resize(f, None, fx=S, fy=S), learningRate=0)
    if i % EVERY == 0:
        masks.append(m.copy())
    i += 1
cap.release()
print(f"掃完 {i} 幀，取樣 {len(masks)} 張遮罩\n", flush=True)

rows = [probe(m) for m in masks]


def q(k, p):
    v = sorted(r[k] for r in rows); return v[min(len(v) - 1, int(p * len(v)))]


print(f"{'':14}{'最小':>10}{'p25':>10}{'中位數':>10}{'p75':>10}{'p95':>10}{'最大':>10}")
for k, lbl in [("n_comp", "元件總數"), ("n_kept", "保留的元件數"),
               ("t_loop", "迴圈版 ms"), ("t_check", "僅檢查 ms"), ("t_vec", "查表版 ms")]:
    print(f"{lbl:14}" + "".join(f"{q(k,p):>10.2f}" for p in (0, .25, .5, .75, .95, .999)))

print(f"\n被門檻刷掉的元件佔比（中位數）："
      f"{(1 - q('n_kept',.5)/q('n_comp',.5))*100:.1f}%")
print(f"整個 Python 檢查迴圈（走訪全部元件）只佔迴圈版的 "
      f"{q('t_check',.5)/q('t_loop',.5)*100:.2f}%")


def corr(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    c = sum((a-mx)*(b-my) for a, b in zip(xs, ys))
    return c / ((sum((a-mx)**2 for a in xs)**.5) * (sum((b-my)**2 for b in ys)**.5))


L = [r["t_loop"] for r in rows]
C = [r["n_comp"] for r in rows]
K = [r["n_kept"] for r in rows]
print(f"\n迴圈版耗時 vs 元件總數      r = {corr(C,L):.3f}")
print(f"迴圈版耗時 vs 保留的元件數   r = {corr(K,L):.3f}   ← 真正的驅動因子")
print(f"元件總數   vs 保留的元件數   r = {corr(C,K):.3f}   ← 兩者本身相關，所以上面第一個 r 是被帶出來的")
print(f"查表版耗時 vs 元件總數      r = {corr(C,[r['t_vec'] for r in rows]):.3f}")
print(f"查表版耗時 vs 保留的元件數   r = {corr(K,[r['t_vec'] for r in rows]):.3f}")
print(f"\n每保留一個元件約多花 "
      f"{st.mean([(r['t_loop']-r['t_check'])/max(1,r['n_kept']) for r in rows])*1000:.0f} 微秒")

json.dump(rows, open("probe_loop_cost.json", "w"), ensure_ascii=False, indent=1)
print("\n明細寫入 probe_loop_cost.json")
