#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""階段 1＋2：在真實影片上量「多久才算背景」與「凍結版會不會漂移」。

一趟解碼、6 組 α 並行。影片解碼（2592x1944）是最貴的一步，
6 組設定共用同一份解碼結果，各自餵給獨立的 KNN 物件，比跑六趟快約 5 倍。

兩個指標同時產出：
  指標 A（階段 1）吸收時間 —— 在地面固定位置貼一塊合成色塊，
        數幾秒後該區域的前景比例掉到初始值的一半以下。
  指標 B（階段 2）背景漂移 —— 逐點記錄全畫面前景比例，
        凍結版預期隨時間往上爬，會更新的版本預期持平。

用法：
    python3 measure_absorption.py --video <path> [--out result.json]
"""
import argparse, json, time, sys
import numpy as np
import cv2

# ── 與線上 sidecar 完全一致的 KNN 參數（codetr_service/server.py）──────────
CM_SCALE = 0.5          # S：半解析度計算
CM_HISTORY = 500        # HIST
CM_DIST2 = 500.0        # DIST
CM_WARM_FRAMES = 150    # WARM：暖機幀數
CM_UPDATE_EVERY = 10    # 每幾幀更新一次背景（10fps → 每秒一次）

# 候選 α。0 = 凍結（＝目前線上行為，也是對照組）
ALPHAS = [0.0, 0.005, 0.002, 0.001, 0.0005, 0.0002]

SAMPLE_EVERY = 5        # 每幾幀取樣一次前景比例
PATCH_FULL = 40         # 合成色塊邊長（全解析度 px）
PROBE_STARTS_MIN = [3.0, 11.0, 19.0]   # 三次探針的起始時間（分）
PROBE_LEN_MIN = 6.0                     # 每次探針的觀察窗長度（分）


def new_knn():
    k = cv2.createBackgroundSubtractorKNN(
        history=CM_HISTORY, dist2Threshold=CM_DIST2, detectShadows=True)
    return k


def pick_patch_sites(cap, n_frames, fps, n_sites):
    """挑 n_sites 個位於『柏油路面』且背景穩定的位置當探針點。

    只用「時間上最穩定」會挑到白色油漆路緣、燒錄時間戳這種靜態但不是路面的地方
    （2026-08-23 第一次跑就是這樣挑歪的），所以要先認出路面：

      – 低飽和（柏油是灰的；樹叢是高飽和綠、護欄是高飽和黃）
      – 中等亮度（濾掉白油漆與深色陰影）
      – 時間標準差低（濾掉車道上車來車往的地方）

    再把合格區域依 y 切成近／中／遠三段，每段各取一點。
    分段是刻意的：同樣大小的垃圾在遠處只佔幾個 px，
    三段分開量才看得出吸收時間會不會隨物體視覺大小改變。
    """
    idxs = np.linspace(0, n_frames - 1, 40).astype(int)
    acc = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            acc.append(cv2.resize(f, None, fx=0.25, fy=0.25))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    stack_bgr = np.stack(acc).astype(np.float32)
    mean_bgr = stack_bgr.mean(axis=0)
    gray = stack_bgr.mean(axis=3)              # (N,h,w)
    std = gray.std(axis=0)
    mean = gray.mean(axis=0)
    hsv = cv2.cvtColor(mean_bgr.astype(np.uint8), cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    h, w = std.shape

    road = (sat < 55) & (mean > 70) & (mean < 190) & (std < 12)
    m = PATCH_FULL // 4 + 6                    # 邊界留白（0.25 縮放後）
    road[:m, :] = road[-m:, :] = False
    road[:, :m] = road[:, -m:] = False
    if road.sum() < 50:
        raise SystemExit("找不到足夠的路面像素，選點條件要放寬")

    ys = np.where(road.any(axis=1))[0]
    y_lo, y_hi = int(ys.min()), int(ys.max())
    edges = np.linspace(y_lo, y_hi + 1, n_sites + 1).astype(int)
    band_names = ["遠", "中", "近"][:n_sites]

    sites = []
    for k in range(n_sites):
        cost = np.where(road, std, 1e9).copy()
        cost[: edges[k], :] = 1e9
        cost[edges[k + 1]:, :] = 1e9
        for s in sites:                        # 與已選點保持距離（全解析度 300px）
            yy, xx = s["y"] // 4, s["x"] // 4
            cost[max(0, yy - 38):yy + 38, max(0, xx - 38):xx + 38] = 1e9
        if cost.min() >= 1e9:
            continue
        y, x = np.unravel_index(np.argmin(cost), cost.shape)
        fy, fx = int(y * 4) - PATCH_FULL // 2, int(x * 4) - PATCH_FULL // 2
        # 與當地背景反差夠大的顏色，確保色差超過 dist2 門檻
        color = (20, 20, 20) if mean[y, x] > 128 else (235, 235, 235)
        sites.append({"x": fx, "y": fy, "color": color, "band": band_names[k],
                      "bg_mean": round(float(mean[y, x]), 1),
                      "bg_sat": round(float(sat[y, x]), 1),
                      "bg_std": round(float(std[y, x]), 2)})
    if len(sites) < n_sites:
        raise SystemExit(f"只挑到 {len(sites)}/{n_sites} 個探針點")
    return sites


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="absorption_result.json")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 幀（除錯用）")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"開不了影片: {args.video}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    W, H = int(cap.get(3)), int(cap.get(4))
    if args.limit:
        n_frames = min(n_frames, args.limit)
    print(f"影片 {args.video}", flush=True)
    print(f"  {n_frames} 幀 / {fps:.1f} fps / {n_frames/fps/60:.1f} 分 / {W}x{H}", flush=True)

    print("挑探針位置（先認出柏油路面，再分近／中／遠三段各取一點）...", flush=True)
    sites = pick_patch_sites(cap, n_frames, fps, len(PROBE_STARTS_MIN))
    for i, s in enumerate(sites):
        print("  探針{} [{}段] ({},{})  亮度 {} 飽和 {} 標準差 {}  色塊亮度 {}".format(
            i + 1, s["band"], s["x"], s["y"], s["bg_mean"], s["bg_sat"],
            s["bg_std"], s["color"][0]), flush=True)

    # 每個 α 一個獨立的 KNN
    models = {a: new_knn() for a in ALPHAS}

    # 探針時間窗（幀）
    probes = []
    for i, tmin in enumerate(PROBE_STARTS_MIN):
        f0 = int(tmin * 60 * fps)
        f1 = int(f0 + PROBE_LEN_MIN * 60 * fps)
        if f1 > n_frames:
            print(f"  ! 探針{i+1} 超出影片長度，跳過", flush=True)
            continue
        probes.append({"i": i, "f0": f0, "f1": f1, "site": sites[i]})

    ps = int(round(PATCH_FULL * CM_SCALE))   # 色塊在半解析度上的邊長
    drift = {a: [] for a in ALPHAS}          # (秒, 全畫面前景比例)
    patch_series = {p["i"]: {a: [] for a in ALPHAS} for p in probes}

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    t0 = time.time()
    last_update = -CM_UPDATE_EVERY

    for idx in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            print(f"  ! 第 {idx} 幀讀不到，提前結束", flush=True)
            n_frames = idx
            break

        # 暖機：所有模型吃一樣的開頭 150 幀，用預設 learningRate
        if idx < CM_WARM_FRAMES:
            small = cv2.resize(frame, None, fx=CM_SCALE, fy=CM_SCALE)
            for m in models.values():
                m.apply(small)
            continue

        active = next((p for p in probes if p["f0"] <= idx < p["f1"]), None)
        if active is not None:
            s = active["site"]
            frame[s["y"]:s["y"] + PATCH_FULL, s["x"]:s["x"] + PATCH_FULL] = s["color"]

        need_update = (idx - last_update) >= CM_UPDATE_EVERY
        need_sample = (idx % SAMPLE_EVERY == 0)
        if not (need_update or need_sample):
            continue

        small = cv2.resize(frame, None, fx=CM_SCALE, fy=CM_SCALE)

        if need_update:
            for a, m in models.items():
                m.apply(small, learningRate=a)   # α=0 → 凍結，等同現行行為
            last_update = idx

        if need_sample:
            sec = idx / fps
            for a, m in models.items():
                mask = m.apply(small, learningRate=0)   # 只查詢，不更新
                fg = (mask == 255)                       # 127 是陰影，不算前景
                drift[a].append((round(sec, 1), round(float(fg.mean()), 6)))
                if active is not None:
                    py = int(active["site"]["y"] * CM_SCALE)
                    px = int(active["site"]["x"] * CM_SCALE)
                    r = float(fg[py:py + ps, px:px + ps].mean())
                    patch_series[active["i"]][a].append((round(sec, 1), round(r, 4)))

        if idx % 2000 == 0 and idx:
            el = time.time() - t0
            print(f"  {idx}/{n_frames} 幀  已耗時 {el/60:.1f} 分  "
                  f"預估總計 {el/idx*n_frames/60:.1f} 分", flush=True)

    cap.release()

    # ── 指標 A：吸收時間 ────────────────────────────────────────────────
    print("\n" + "=" * 78, flush=True)
    print("指標 A — 吸收時間：色塊貼上後，多久其前景比例掉到初始值的一半以下", flush=True)
    print("=" * 78, flush=True)
    hdr = f"{'α':>9} {'名目 0.19/α':>12} " + " ".join(
        f"{'探針'+str(p['i']+1)+'('+str(int(PROBE_STARTS_MIN[p['i']]))+'分)':>13}" for p in probes)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    absorb = {}
    for a in ALPHAS:
        nominal = "凍結" if a == 0 else f"{0.19/a:.0f} 秒"
        row, cells = {}, []
        for p in probes:
            ser = patch_series[p["i"]][a]
            if not ser:
                cells.append(f"{'-':>13}"); continue
            base = np.mean([v for _, v in ser[:4]])   # 貼上後最初 2 秒當基準
            t_start = ser[0][0]
            hit = next((t for t, v in ser if v < base * 0.5), None)
            if hit is None:
                cells.append(f"{'>'+str(int(PROBE_LEN_MIN*60))+'秒':>13}")
                row[f"probe{p['i']+1}"] = None
            else:
                cells.append(f"{hit - t_start:>10.0f} 秒")
                row[f"probe{p['i']+1}"] = round(hit - t_start, 1)
        absorb[str(a)] = row
        print(f"{a:>9} {nominal:>12} " + " ".join(cells), flush=True)
    print("\n  註：'>360秒' 表示在 6 分鐘觀察窗內都沒被吸收；基準值取貼上後最初 2 秒的平均。", flush=True)

    # ── 指標 B：背景漂移 ────────────────────────────────────────────────
    print("\n" + "=" * 78, flush=True)
    print("指標 B — 背景漂移：全畫面前景比例，前 10 分鐘平均 vs 最後 10 分鐘平均", flush=True)
    print("=" * 78, flush=True)
    print(f"{'α':>9} {'前10分':>10} {'後10分':>10} {'變化':>10}   判讀", flush=True)
    print("-" * 62, flush=True)
    total_sec = n_frames / fps
    drift_sum = {}
    for a in ALPHAS:
        arr = drift[a]
        early = [v for t, v in arr if t <= 600]
        late = [v for t, v in arr if t >= total_sec - 600]
        e, l = float(np.mean(early)), float(np.mean(late))
        ratio = l / e if e > 0 else float("nan")
        verdict = "往上爬 ←問題" if ratio > 1.3 else ("持平" if ratio < 1.15 else "略升")
        drift_sum[str(a)] = {"early": round(e, 6), "late": round(l, 6),
                             "ratio": round(ratio, 3)}
        print(f"{a:>9} {e*100:>9.3f}% {l*100:>9.3f}% {ratio:>9.2f}x   {verdict}", flush=True)

    out = {
        "video": args.video, "n_frames": n_frames, "fps": fps, "size": [W, H],
        "params": {"scale": CM_SCALE, "history": CM_HISTORY, "dist2": CM_DIST2,
                   "warm": CM_WARM_FRAMES, "update_every": CM_UPDATE_EVERY,
                   "patch_full_px": PATCH_FULL, "sample_every": SAMPLE_EVERY},
        "sites": sites, "absorb": absorb, "drift_summary": drift_sum,
        "drift_series": {str(a): drift[a] for a in ALPHAS},
        "elapsed_min": round((time.time() - t0) / 60, 1),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n完整資料寫入 {args.out}（含逐點漂移曲線，可畫圖）", flush=True)
    print(f"總耗時 {out['elapsed_min']} 分", flush=True)


if __name__ == "__main__":
    main()
