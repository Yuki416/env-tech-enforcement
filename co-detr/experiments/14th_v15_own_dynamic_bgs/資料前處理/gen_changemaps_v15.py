#!/usr/bin/env python3
"""Step 2-3（第 14 次實驗 / v15）：用「事件前 pre-roll 凍結背景」的兩段式 KNN 生成 changemap。

與第 11/12 次（gen_changemaps_v13.py）的差異
------------------------------------------
- 第 11/12 次的 **test 分支**是「整片暖機」（拿含落地靜止段的整部 clip 建背景）→ 落地垃圾被學進背景、
  在 changemap 上消失。實測我們的 594 靜止 test 有 74% 的 litter 框全黑。
- v15 **所有 split 一律用 pre-roll 暖機**：只餵 clip 開始*之前*的 WARM 幀建「乾淨地面」背景，凍結後
  對 dataset 每張圖取前景 → 落地垃圾相對乾淨地面仍是前景。這也是因果的（pre-roll 全是過去幀，可部署）。

方法（鎖定，沿用第 11/12 次已驗證的 KNN 參數）
    KNN(history=500, dist2Threshold=500, detectShadows=True)
    pass1：只餵 .avi[warm_start:first_abs] 的 WARM 幀（不含任何要用的幀）建背景。
    pass2：對 dataset jpg 每張 knn.apply(img, learningRate=0)（凍結背景）取前景。
    後處理：rm5 移除 <5px 碎塊，並把 shadow(127) 濾成 0 → 嚴格二值 {0,255}。
    half-res 計算 → INTER_NEAREST 升回全解析度存平行灰階 PNG（同檔名 .png）。

對齊：讀 alignment_v15.json（build_alignment_v15.py 產出）。first_abs = clip 首幀在 .avi 的絕對索引。
      **pass2 算在 dataset jpg 上**，故 changemap 與模型看到的 RGB 天然逐像素對齊；.avi 只供暖機。

edge case：pre-roll=0（20240713054438）→ 無事件前 pre-roll。人工覆寫改用「尾巴暖機」(method C)：
    .avi 全 201 幀、用幀=開頭 15 幀(動態 litter)，尾巴 frame 15~200(垃圾落地後)當暖機素材建背景。
    移動中的垃圾在查詢幀位置一直變 → 用落地後尾巴凍結背景查詢仍為前景（probe_edge_clip.py 已驗證框內 0.4~0.9）。
    見 TAIL_WARMUP。其餘 64 部一律 pre-roll。

輸出：<DST>/{split}_changemap/<同 jpg 檔名>.png。可重跑（跳過已存）。
"""
import json, re, sys, collections
from pathlib import Path
import cv2, numpy as np

V14 = Path("/path/to/data/coco/chiayi_v14_own_dynamic_litter")
DST = Path("/path/to/data/coco/chiayi_v15_own_dynamic_bgs")
ALIGN = Path(__file__).resolve().parent / "alignment_v15.json"
S = 0.5           # half-res 計算
HIST, DIST = 500, 500.0
RM = 5            # 移除小於 RM 像素的前景碎塊

# 人工覆寫：0 pre-roll 的 clip 改用「尾巴暖機」(method C)。key = <split>/<ts>_<MMmSSs>。
# 餵 .avi[start:end)（用幀之後的落地靜止段）建背景，不含任何查詢幀。
TAIL_WARMUP = {
    "train/20240713054438_00m00s": dict(start=15, end=201),   # .avi 全 201 幀，用幀=abs 0~14
}

def parse(name):
    g = re.search(r"(\d{14})_(\d{2})m(\d{2})s", name)
    return g.group(1), int(g.group(2)) * 60 + int(g.group(3))

def rm5(mask):
    """只保留 ==255 的硬前景（濾掉 shadow=127），移除 <RM 像素碎塊，輸出 {0,255}。"""
    fg = (mask == 255).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    out = np.zeros_like(fg)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= RM:
            out[lab == i] = 1
    return (out * 255).astype(np.uint8)

def main():
    align = json.load(open(ALIGN))
    want = sys.argv[1:] or ["train", "valid", "test"]   # 可指定 split，例：python gen_changemaps_v15.py test
    # 依 (split, ts, offs) 收 dataset 影像
    clips = collections.defaultdict(list)
    for split in want:
        d = json.load(open(V14 / f"annotations/{split}.json"))
        for im in d["images"]:
            ts, offs = parse(im["file_name"])
            clips[(split, ts, offs)].append(im["file_name"])

    summary = collections.Counter()
    for (split, ts, offs), names in sorted(clips.items()):
        key = f"{split}/{ts}_{offs//60:02d}m{offs%60:02d}s"
        info = align[key]
        outdir = DST / f"{split}_changemap"
        outdir.mkdir(parents=True, exist_ok=True)

        # ── pass1：暖機（一般 = pre-roll；TAIL_WARMUP 覆寫 = 尾巴）────────
        knn = cv2.createBackgroundSubtractorKNN(history=HIST, dist2Threshold=DIST, detectShadows=True)
        cap = cv2.VideoCapture(info["avi"])
        if key in TAIL_WARMUP:
            warm_range = range(TAIL_WARMUP[key]["start"], TAIL_WARMUP[key]["end"])
            warm_tag = f"尾巴暖機[{warm_range.start}:{warm_range.stop}]"
        else:
            warm_range = range(info["warm_start"], info["first_abs"])   # pre-roll
            warm_tag = ""
        n_warm = 0
        if len(warm_range) > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, warm_range.start)
            for _ in warm_range:
                ok, f = cap.read()
                if not ok:
                    break
                knn.apply(cv2.resize(f, None, fx=S, fy=S))
                n_warm += 1
        cap.release()

        # ── pass2：對 dataset jpg 凍結取前景 ────────────────────
        saved = skipped = 0
        for name in names:
            outp = outdir / (name[:-4] + ".png")
            if outp.exists():
                skipped += 1
                continue
            img = cv2.imread(str(V14 / split / name))
            H, W = img.shape[:2]
            if n_warm == 0:
                mask_full = np.zeros((H, W), np.uint8)      # 無暖機 → 全黑
            else:
                fg = knn.apply(cv2.resize(img, None, fx=S, fy=S), learningRate=0)
                mask_full = cv2.resize(rm5(fg), (W, H), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(str(outp), mask_full)
            saved += 1

        flag = ""
        if warm_tag:
            flag = f"  ★ {warm_tag}"
        elif n_warm == 0:
            flag = "  ⚠️ 無暖機→全黑"; summary["全黑clip"] += 1
        elif info["preroll"] < 150:
            flag = f"  (pre-roll={info['preroll']})"
        summary["clip"] += 1; summary["saved"] += saved; summary["skipped"] += skipped
        print(f"[{split:5s}] {ts} {offs//60:02d}m{offs%60:02d}s  暖機{n_warm}幀 → "
              f"存{saved} 跳過{skipped} / 共{len(names)}{flag}")
        sys.stdout.flush()

    print(f"\n✅ 完成：{summary['clip']} clips，新存 {summary['saved']} 張，"
          f"跳過 {summary['skipped']} 張，全黑 clip {summary['全黑clip']} 部。")

if __name__ == "__main__":
    main()
