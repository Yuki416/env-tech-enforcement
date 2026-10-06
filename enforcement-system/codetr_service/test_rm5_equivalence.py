#!/usr/bin/env python3
"""驗證 rm5 迴圈版與查表版輸出逐像素相同（邊界情況 + 真實遮罩）。

優化_rm5向量化.md 第 6 節引用這支腳本的輸出。
先前那一節寫「另外測，全部通過」但沒有腳本可指，等於無法重跑，這支補上。

不需要 GPU。邊界情況部分不讀影片，秒級完成；加 --video 才會另外跑真實遮罩。
用法：
    python3 test_rm5_equivalence.py            # 只跑邊界情況（約 1 秒）
    python3 test_rm5_equivalence.py --video    # 另外跑基準影片 100 張遮罩（約 4 分鐘）
"""
import argparse, sys
import cv2, numpy as np

RM = 5
AVI = ("/path/to/data/raw_videos/7月份影片/113.7.7 OK/"
       "case57/"
       "537_8_loc537_20240707063733.avi")
S, HIST, DIST, WARM, SAMPLE_EVERY = 0.5, 500, 500.0, 150, 150


def rm5_loop(mask):
    """gen_changemaps_v18.py:51 的寫法（訓練資料生成用的基準）。"""
    fg = (mask == 255).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    out = np.zeros_like(fg)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= RM:
            out[lab == i] = 1
    return (out * 255).astype(np.uint8)


def rm5_vec(mask):
    """server.py:84 的寫法（線上服務用的）。"""
    fg = (mask == 255).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    keep = np.zeros(n, dtype=np.uint8)
    keep[1:] = (st[1:, cv2.CC_STAT_AREA] >= RM)
    return (keep[lab] * 255).astype(np.uint8)


def n_comp(m):
    n, _, _, _ = cv2.connectedComponentsWithStats((m == 255).astype(np.uint8), 8)
    return n - 1


def edge_cases():
    """每個 case 是 (名稱, 遮罩, 預期保留的前景像素數)。"""
    cases = []

    cases.append(("全黑（沒有任何前景）", np.zeros((40, 40), np.uint8), 0))
    cases.append(("全白（單一巨大元件）", np.full((40, 40), 255, np.uint8), 1600))
    cases.append(("整張都是陰影 127", np.full((40, 40), 127, np.uint8), 0))

    a = np.zeros((40, 40), np.uint8); a[5:7, 5:7] = 255          # 2x2 = 4 px < RM
    cases.append(("只有 4 像素碎塊（應全刪）", a, 0))

    b = a.copy(); b[20, 10:15] = 255                              # 再加一條 5 px
    cases.append(("4 像素碎塊＋5 像素元件（應只留後者）", b, 5))

    c = np.zeros((40, 40), np.uint8); c[10, 10:15] = 255          # 剛好等於門檻
    cases.append(("剛好 5 像素（門檻是 >=，應保留）", c, 5))

    d = np.zeros((40, 40), np.uint8); d[10, 10:14] = 255          # 剛好差一個
    cases.append(("剛好 4 像素（應刪除）", d, 0))

    # 斜角相連：八方向連通算同一塊（4 px），四方向會算成 2 塊。確認走的是八方向。
    e = np.zeros((40, 40), np.uint8); e[10, 10] = e[10, 11] = e[11, 12] = e[11, 13] = 255
    cases.append(("斜角相連 4 像素（八方向算一塊，<5 應刪）", e, 0))

    # 前景貼齊影像邊界，確認沒有 off-by-one
    f = np.zeros((40, 40), np.uint8); f[0, 0:6] = 255
    cases.append(("貼齊左上角邊界的 6 像素元件", f, 6))

    # 元件數遠超過 255：確認 uint8 查表的「值」不會溢位（索引是 int32）
    g = np.zeros((200, 200), np.uint8); g[::4, ::4] = 255         # 2500 個 1px 碎塊，全該刪
    g[100:110, 100:110] = 255                                     # 一個 100px 元件，該留
    cases.append((f"{n_comp(g)} 個元件（>255，測 uint8 查表溢位）", g, 100))

    # 255 與 127 混在同一塊：127 必須先被丟掉，不能併進元件
    h = np.zeros((40, 40), np.uint8); h[10, 10:13] = 255; h[10, 13:20] = 127
    cases.append(("3 像素前景旁邊接 7 像素陰影（陰影不算面積，應刪）", h, 0))

    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="store_true", help="另外跑基準影片的 100 張真實遮罩")
    a = ap.parse_args()
    print(f"cv2 {cv2.__version__} / numpy {np.__version__}　CM_MIN_AREA={RM}\n")

    bad = 0
    print("=== 邊界情況 ===")
    for name, m, want in edge_cases():
        lo, ve = rm5_loop(m), rm5_vec(m)
        same = np.array_equal(lo, ve)
        got = int((ve == 255).sum())
        ok = same and got == want
        bad += not ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:44} 兩版相同={same} "
              f"保留前景={got}（預期 {want}）")

    if a.video:
        print(f"\n=== 真實遮罩（基準影片全片，每 {SAMPLE_EVERY} 幀取一張）===")
        cap = cv2.VideoCapture(AVI)
        if not cap.isOpened():
            print(f"  [SKIP] 開不了影片 {AVI}")
        else:
            knn = cv2.createBackgroundSubtractorKNN(history=HIST, dist2Threshold=DIST,
                                                    detectShadows=True)
            for _ in range(WARM):
                ok, f = cap.read()
                knn.apply(cv2.resize(f, None, fx=S, fy=S))
            i, n_same, n_tot = 0, 0, 0
            while True:
                ok, f = cap.read()
                if not ok:
                    break
                m = knn.apply(cv2.resize(f, None, fx=S, fy=S), learningRate=0)
                if i % SAMPLE_EVERY == 0:
                    n_tot += 1
                    n_same += np.array_equal(rm5_loop(m), rm5_vec(m))
                i += 1
            cap.release()
            bad += (n_same != n_tot)
            print(f"  [{'PASS' if n_same == n_tot else 'FAIL'}] "
                  f"逐像素相同 {n_same}/{n_tot} 張")

    print(f"\n{'全部通過' if bad == 0 else f'★ {bad} 項失敗 ★'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
