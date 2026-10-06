#!/usr/bin/env python3
"""bench_rm5.py 的加大版：遮罩取樣涵蓋整支影片，並多測兩個微優化變體。

第一版只取暖機後連續 20 幀，那 20 幀剛好都在同一個交通狀況裡，
元件數的分佈不具代表性（中位數會隨取樣位置大幅移動）。
這一版每 150 幀（15 秒）記錄一張遮罩，涵蓋整支 25 分鐘影片。

影片：537_8_loc537_20240707063733.avi
（交付資料夾叫「案57」，但 6/7/8/9 月批次各有一個案57、指向四支不同影片，一律用檔名。）

變體：
  loop    迴圈版 —— 每個元件對整張圖做一次 `lab == i`
  vec     查表版 —— keep[lab] 再乘 255（目前 server.py 的寫法）
  vec255  查表版但查表本身就存 255，省掉最後那一次全圖乘法
  vec_cmp vec255 再加上用 cv2.compare 取代 (mask==255).astype()
"""
import time, json, statistics as stats
import cv2, numpy as np

AVI = ("/path/to/data/raw_videos/7月份影片/113.7.7 OK/"
       "case57/"
       "537_8_loc537_20240707063733.avi")
S, HIST, DIST, RM, WARM = 0.5, 500, 500.0, 5, 150
SAMPLE_EVERY, REPEAT = 150, 2


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


def rm5_vec255(mask):
    fg = (mask == 255).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    keep = np.zeros(n, dtype=np.uint8)
    keep[1:] = np.where(st[1:, cv2.CC_STAT_AREA] >= RM, 255, 0)
    return keep[lab]


def rm5_vec_cmp(mask):
    fg = cv2.compare(mask, 255, cv2.CMP_EQ)      # 直接得到 {0,255} uint8
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    keep = np.zeros(n, dtype=np.uint8)
    keep[1:] = np.where(st[1:, cv2.CC_STAT_AREA] >= RM, 255, 0)
    return keep[lab]


def cc_only(mask):
    return cv2.connectedComponentsWithStats((mask == 255).astype(np.uint8), 8)


VARIANTS = [("loop", rm5_loop), ("vec", rm5_vec), ("vec255", rm5_vec255),
            ("vec_cmp", rm5_vec_cmp), ("cc_only", cc_only)]


def timeit(fn, arg, repeat=REPEAT):
    best = float("inf")
    for _ in range(repeat):
        t = time.perf_counter(); fn(arg); best = min(best, time.perf_counter() - t)
    return best * 1e3


print(f"cv2 {cv2.__version__} / numpy {np.__version__}", flush=True)
cap = cv2.VideoCapture(AVI)
fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"影片 {total} 幀 / {fps} fps = {total/fps/60:.1f} 分，每 {SAMPLE_EVERY} 幀取一張遮罩", flush=True)

knn = cv2.createBackgroundSubtractorKNN(history=HIST, dist2Threshold=DIST, detectShadows=True)
for _ in range(WARM):
    ok, f = cap.read()
    if not ok:
        raise SystemExit("暖機讀不到幀")
    knn.apply(cv2.resize(f, None, fx=S, fy=S))
print("暖機完成，開始掃全片", flush=True)

masks, idx, t0 = [], WARM, time.perf_counter()
while True:
    ok, f = cap.read()
    if not ok:
        break
    m = knn.apply(cv2.resize(f, None, fx=S, fy=S), learningRate=0)
    if (idx - WARM) % SAMPLE_EVERY == 0:
        masks.append(m.copy())
    idx += 1
cap.release()
print(f"掃完 {idx-WARM} 幀（{time.perf_counter()-t0:.0f}s），取樣 {len(masks)} 張遮罩\n", flush=True)

rows = []
for i, m in enumerate(masks):
    n, lab, st, _ = cv2.connectedComponentsWithStats((m == 255).astype(np.uint8), 8)
    ref = rm5_loop(m)
    r = dict(i=i, frame=WARM + i * SAMPLE_EVERY, n_comp=n - 1,
             n_kept=int((st[1:, cv2.CC_STAT_AREA] >= RM).sum()) if n > 1 else 0,
             fg_pct=float((m == 255).sum()) / m.size * 100)
    for name, fn in VARIANTS:
        r[name] = timeit(fn, m)
        if name not in ("loop", "cc_only"):
            r["same_" + name] = bool(np.array_equal(fn(m), ref))
    rows.append(r)
    if i % 10 == 0:
        print(f"  #{i:03d} 幀{r['frame']:6d} 元件{r['n_comp']:6d} "
              f"loop {r['loop']:7.1f} vec {r['vec']:5.2f} vec255 {r['vec255']:5.2f} "
              f"vec_cmp {r['vec_cmp']:5.2f} cc {r['cc_only']:5.2f} ms", flush=True)

def q(k, p):
    v = sorted(r[k] for r in rows)
    return v[min(len(v) - 1, int(p * len(v)))]

print(f"\n=== {len(rows)} 張遮罩 · 半解析度 1296x972 ===")
print(f"{'':16}{'最小':>10}{'p25':>10}{'中位數':>10}{'p75':>10}{'p95':>10}{'最大':>10}")
for k, lbl in [("n_comp", "連通元件數"), ("fg_pct", "前景比 %"),
               ("loop", "迴圈版 ms"), ("vec", "查表版 ms"),
               ("vec255", "查表255 ms"), ("vec_cmp", "查表cmp ms"),
               ("cc_only", "CC 本身 ms")]:
    print(f"{lbl:16}" + "".join(f"{q(k,p):>10.2f}" for p in (0, .25, .5, .75, .95, .999)))

print(f"\n加速倍數（迴圈 ÷ 查表）")
for p, lbl in [(.25, "p25"), (.5, "中位數"), (.75, "p75"), (.95, "p95"), (.999, "最大")]:
    print(f"  {lbl:8} {q('loop',p)/q('vec',p):6.1f} 倍")
print(f"  全片總計 {sum(r['loop'] for r in rows)/sum(r['vec'] for r in rows):.1f} 倍"
      f"（迴圈 {sum(r['loop'] for r in rows)/1000:.1f}s vs 查表 {sum(r['vec'] for r in rows)/1000:.1f}s）")

for name in ("vec", "vec255", "vec_cmp"):
    ok = sum(r["same_" + name] for r in rows)
    print(f"逐像素與迴圈版相同：{name:8} {ok}/{len(rows)}")

xs = [r['n_comp'] for r in rows]; ys = [r['loop'] - r['cc_only'] for r in rows]
mx, my = stats.mean(xs), stats.mean(ys)
cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
sx = sum((x-mx)**2 for x in xs)**.5; sy = sum((y-my)**2 for y in ys)**.5
print(f"\n迴圈過濾耗時 vs 元件數：r = {cov/(sx*sy):.3f}，每元件約 {my/mx*1000:.1f} 微秒")
zs = [r['vec'] - r['cc_only'] for r in rows]
mz = stats.mean(zs); cov2 = sum((x-mx)*(z-mz) for x, z in zip(xs, zs))
sz = sum((z-mz)**2 for z in zs)**.5
print(f"查表過濾耗時 vs 元件數：r = {cov2/(sx*sz):.3f}（應接近 0：與元件數無關）")

json.dump(rows, open("bench_rm5_full.json", "w"), ensure_ascii=False, indent=1)
print("\n明細寫入 bench_rm5_full.json")
