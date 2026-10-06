#!/usr/bin/env python3
"""量 rm5 兩種寫法的速度差，並驗證輸出逐像素相同。

rm5 = changemap 後處理：把 KNN 前景遮罩裡面積 <5px 的連通元件刪掉。
兩種寫法算的是同一件事，只差在「怎麼把保留清單畫回圖上」：

  迴圈版（訓練資料生成 gen_changemaps_v18.py 用的）：
      for i in 1..n-1: if 面積夠大: out[lab == i] = 1
      → 每個元件對整張圖做一次布林比較，共 n-1 次全圖掃描。

  查表版（server.py 現在用的）：
      keep = 每個元件編號要不要留（長度 n 的陣列）
      out  = keep[lab]                      → 一次全圖索引。

用真實影片的遮罩量，不是合成資料。影片：537_8_loc537_20240707063733.avi
（交付資料夾叫「案57」，但 6/7/8/9 月批次各有一個案57、指向四支不同影片，所以一律用檔名。）
"""
import time, json, statistics as stats
import cv2, numpy as np

AVI = ("/path/to/data/raw_videos/7月份影片/113.7.7 OK/"
       "case57/"
       "537_8_loc537_20240707063733.avi")
S, HIST, DIST, RM, WARM = 0.5, 500, 500.0, 5, 150
N_MASK, REPEAT = 20, 3


def rm5_loop(mask):
    fg = (mask == 255).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    out = np.zeros_like(fg)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= RM:
            out[lab == i] = 1
    return (out * 255).astype(np.uint8)


def rm5_vec(mask):
    fg = (mask == 255).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    keep = np.zeros(n, dtype=np.uint8)
    keep[1:] = (st[1:, cv2.CC_STAT_AREA] >= RM)
    return (keep[lab] * 255).astype(np.uint8)


def timeit(fn, arg, repeat):
    """回傳最快一次的秒數。取最小值而不是平均，避免作業系統雜訊灌水。"""
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter()
        fn(arg)
        best = min(best, time.perf_counter() - t)
    return best


print(f"cv2 {cv2.__version__} / numpy {np.__version__}", flush=True)
cap = cv2.VideoCapture(AVI)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"影片 fps={fps}", flush=True)

knn = cv2.createBackgroundSubtractorKNN(history=HIST, dist2Threshold=DIST, detectShadows=True)
t0 = time.perf_counter()
for _ in range(WARM):
    ok, f = cap.read()
    if not ok:
        raise SystemExit("暖機讀不到幀")
    knn.apply(cv2.resize(f, None, fx=S, fy=S))
print(f"暖機 {WARM} 幀完成（{time.perf_counter()-t0:.1f}s）", flush=True)

masks, decode_ms, resize_ms, apply_ms, full_hw = [], [], [], [], None
for _ in range(N_MASK):
    t = time.perf_counter(); ok, f = cap.read(); decode_ms.append((time.perf_counter()-t)*1e3)
    if not ok:
        break
    full_hw = f.shape[:2]
    t = time.perf_counter(); small = cv2.resize(f, None, fx=S, fy=S); resize_ms.append((time.perf_counter()-t)*1e3)
    t = time.perf_counter(); m = knn.apply(small, learningRate=0); apply_ms.append((time.perf_counter()-t)*1e3)
    masks.append(m)
cap.release()
print(f"取得 {len(masks)} 張遮罩，半解析度 {masks[0].shape[1]}x{masks[0].shape[0]}"
      f"（原始 {full_hw[1]}x{full_hw[0]}）\n", flush=True)

rows = []
for i, m in enumerate(masks):
    fgpx = int((m == 255).sum())
    n, lab, st, _ = cv2.connectedComponentsWithStats((m == 255).astype(np.uint8), 8)
    kept = int((st[1:, cv2.CC_STAT_AREA] >= RM).sum()) if n > 1 else 0
    a = rm5_loop(m); b = rm5_vec(m)
    same = bool(np.array_equal(a, b))
    tl = timeit(rm5_loop, m, REPEAT) * 1e3
    tv = timeit(rm5_vec, m, REPEAT) * 1e3
    # 只算 connectedComponentsWithStats 本身，看兩版共用的成本佔多少
    tc = timeit(lambda x: cv2.connectedComponentsWithStats((x == 255).astype(np.uint8), 8), m, REPEAT) * 1e3
    rows.append(dict(i=i, fg_px=fgpx, fg_pct=fgpx / m.size * 100, n_comp=n - 1, n_kept=kept,
                     loop_ms=tl, vec_ms=tv, cc_ms=tc, identical=same))
    print(f"  #{i:02d} 元件 {n-1:6d}（留 {kept:5d}） 前景 {fgpx/m.size*100:5.2f}%  "
          f"迴圈 {tl:7.1f} ms  查表 {tv:5.2f} ms  其中 CC {tc:5.2f} ms  "
          f"{'相同' if same else '★不同★'}", flush=True)

med = lambda k: stats.median(r[k] for r in rows)
print("\n=== 中位數 ===")
print(f"連通元件數        {med('n_comp'):.0f}")
print(f"保留的元件數      {med('n_kept'):.0f}")
print(f"前景比            {med('fg_pct'):.2f}%")
print(f"迴圈版            {med('loop_ms'):.1f} ms")
print(f"查表版            {med('vec_ms'):.2f} ms")
print(f"  其中 CC 本身    {med('cc_ms'):.2f} ms（兩版共用，不可省）")
print(f"  查表那一步      {med('vec_ms')-med('cc_ms'):.2f} ms")
print(f"  迴圈那一步      {med('loop_ms')-med('cc_ms'):.1f} ms")
print(f"加速              {med('loop_ms')/med('vec_ms'):.1f} 倍（整個 rm5）"
      f" / {(med('loop_ms')-med('cc_ms'))/max(1e-9, med('vec_ms')-med('cc_ms')):.0f} 倍（只看過濾那一步）")
print(f"逐像素相同        {sum(r['identical'] for r in rows)}/{len(rows)}")

print("\n=== 每幀其他成本（中位數，同一批幀）===")
print(f"解碼            {stats.median(decode_ms):.1f} ms")
print(f"縮成半解析度    {stats.median(resize_ms):.1f} ms")
print(f"KNN apply       {stats.median(apply_ms):.2f} ms")

# 迴圈成本是否隨元件數線性成長
xs = [r['n_comp'] for r in rows]; ys = [r['loop_ms'] - r['cc_ms'] for r in rows]
mx, my = stats.mean(xs), stats.mean(ys)
cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
sx = sum((x-mx)**2 for x in xs) ** .5; sy = sum((y-my)**2 for y in ys) ** .5
print(f"\n過濾步驟耗時 vs 元件數  相關係數 r = {cov/(sx*sy):.3f}"
      f"（每個元件約 {my/mx*1000:.1f} 微秒）")

json.dump(rows, open("bench_rm5.json", "w"), ensure_ascii=False, indent=1)
print("\n明細寫入 bench_rm5.json")
