#!/usr/bin/env python3
"""量「訓練 stride=1」與「線上 stride=35」實際造成什麼差別。

三個問題：
  Q1 同一個 KNN session，背景模型隨著 apply 次數漂多遠？（訓練只 apply ~22 次就結束，線上 ~430 次）
  Q2 同一段影片內容，stride=1 與 stride=35 各自的 changemap 前景比是多少？
  Q3 訓練那種「session 只涵蓋 2.2 秒」的設定下，背景等於暖機那 150 幀的靜止照嗎？

用真實影片 537_8_loc537_20240707063733.avi。所有數字都是實測。
（交付資料夾叫「案57」，但該編號在 6/8/9 月批次各指向不同影片，引用請用檔名。）
"""
import cv2, numpy as np, json, time

AVI = ("/path/to/data/raw_videos/7月份影片/113.7.7 OK/"
       "case57/"
       "537_8_loc537_20240707063733.avi")
S, HIST, DIST, WARM = 0.5, 500, 500.0, 150

def new_knn():
    return cv2.createBackgroundSubtractorKNN(history=HIST, dist2Threshold=DIST,
                                             detectShadows=True)

def warm(knn, cap, n=WARM):
    for _ in range(n):
        ok, f = cap.read()
        if not ok: break
        knn.apply(cv2.resize(f, None, fx=S, fy=S))

cap = cv2.VideoCapture(AVI)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"影片 fps={fps} 總幀={total} 長度={total/fps/60:.1f} 分", flush=True)

# ---------- 跑兩條 session：stride=1（訓練式）與 stride=35（線上式） ----------
# 兩條都從同一個暖機狀態出發，之後只差「餵哪些幀」。
res = {}
for stride, label in [(1, "訓練式 stride=1"), (35, "線上式 stride=35")]:
    cap = cv2.VideoCapture(AVI)
    knn = new_knn()
    warm(knn, cap)
    bg0 = knn.getBackgroundImage().astype(np.float32)
    fg_ratio, drift, napply = [], [], 0
    idx = WARM
    t0 = time.monotonic()
    while True:
        ok, f = cap.read()
        if not ok: break
        idx += 1
        if (idx - WARM) % stride: continue
        m = knn.apply(cv2.resize(f, None, fx=S, fy=S), learningRate=0)
        napply += 1
        fg_ratio.append(float((m == 255).mean()))
        if napply in (1, 6, 22, 100, 430) or napply % 200 == 0:
            bg = knn.getBackgroundImage().astype(np.float32)
            drift.append((napply, idx, float(np.abs(bg - bg0).mean())))
    res[label] = dict(stride=stride, napply=napply,
                      fg_mean=float(np.mean(fg_ratio)),
                      fg_med=float(np.median(fg_ratio)),
                      drift=drift, sec=time.monotonic() - t0)
    print(f"{label}: apply {napply} 次, 前景比 mean={np.mean(fg_ratio)*100:.2f}% "
          f"median={np.median(fg_ratio)*100:.2f}%, 耗時 {res[label]['sec']:.0f}s", flush=True)
    for na, fi, d in drift:
        print(f"    apply#{na:<5d} 影片幀{fi:<7d} 背景與暖機的平均像素差 {d:6.2f}", flush=True)

json.dump(res, open("/tmp/stride_gap.json", "w"),
          ensure_ascii=False, indent=1)
print("done", flush=True)
