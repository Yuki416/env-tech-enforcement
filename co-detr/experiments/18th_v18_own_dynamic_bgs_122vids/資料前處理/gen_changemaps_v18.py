#!/usr/bin/env python3
"""第18次 / v18：用「事件前 pre-roll 凍結背景」兩段式 KNN 生成 changemap，寫進 v18 資料集，
並產出「逐 clip 處理記錄 MD」＋ sanity（使用者要求：每部 clip 如何處理都要逐一記錄）。

方法（鎖定，沿用第 11/12/14 次已驗證）
    KNN(history=500, dist2Threshold=500, detectShadows=True)
    pass1：只餵 .avi[warm_start:first_abs] 的 pre-roll 幀（事件前、不含任何要用的幀）建乾淨地面背景。
    pass2：對 dataset jpg 每張 knn.apply(img, learningRate=0) 凍結取前景。
    後處理：rm5 移除 <5px 碎塊、shadow(127)→0，嚴格二值 {0,255}；half-res 算 → INTER_NEAREST 升全解析度。
    對齊：讀 alignment_v18.json（build_alignment_v18.py 產出）。

edge case：pre-roll=0（train/20240713054438_00m00s，同 v15）→ 尾巴暖機 method C：
    餵 .avi[15:201]（用幀 1~15 之後的落地靜止段）建背景。v15 已驗證框內前景 0.4~0.9。

輸出：
  - v18/{split}_changemap/<同 jpg 檔名>.png（可重跑，已存則讀回算 sanity）
  - changemap_逐clip處理記錄_v18.md（每 clip 一列：窗口/手動/暖機/WARM幀/框內前景比/全黑幀率/備註）
"""
import json, re, sys, collections
from pathlib import Path
import cv2, numpy as np

HERE  = Path(__file__).resolve().parent
V18   = Path("/path/to/data/coco/chiayi_v18_own_dynamic_bgs_122vids")
ALIGN = HERE / "alignment_v18.json"
MD    = HERE / "changemap_逐clip處理記錄_v18.md"
S = 0.5
HIST, DIST = 500, 500.0
RM = 5
WARM = 150
LITTER, GARBAGE = 2, 5   # category_id

# 0-preroll clip 的尾巴暖機（method C）。key = <split>/<ts>_<MMmSSs>。（同 v15，已驗證）
TAIL_WARMUP = {
    "train/20240713054438_00m00s": dict(start=15, end=201),
}
# 8 部手動 RANGE_OVERRIDE clip（供 MD 標記「手動 vs HEAD」）。key = <ts>_<MMmSSs>
MANUAL = {
    "20240828064620_00m12s", "20240603181958_04m49s", "20240826063130_00m27s",
    "20240906054955_01m00s", "20240802173010_00m54s", "20240804171613_24m02s",
    "20240911175753_04m39s", "20240803182230_06m22s",
}

def parse(name):
    g = re.search(r"(\d{14})_(?:(\d{2})h)?(\d{2})m(\d{2})s", name)
    h = int(g.group(2)) if g.group(2) else 0
    offs = h * 3600 + int(g.group(3)) * 60 + int(g.group(4))
    fn = int(re.search(r"_frame_(\d+)", name).group(1))
    return g.group(1), offs, fn

def rm5(mask):
    fg = (mask == 255).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, 8)
    out = np.zeros_like(fg)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] >= RM:
            out[lab == i] = 1
    return (out * 255).astype(np.uint8)

def box_fg_ratio(cm, bbox):
    x, y, w, h = [int(round(v)) for v in bbox]
    H, W = cm.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    sub = cm[y0:y1, x0:x1]
    return float((sub > 0).mean())

def main():
    align = json.load(open(ALIGN))
    want = sys.argv[1:] or ["train", "valid", "test"]

    # 收 dataset 影像 + GT（litter/garbage 框）
    clips = collections.defaultdict(list)          # (split,ts,offs) -> [file_name]
    gt = collections.defaultdict(lambda: collections.defaultdict(list))  # split -> fn -> [(cat,bbox)]
    for split in want:
        d = json.load(open(V18 / f"annotations/{split}.json"))
        id2fn = {im["id"]: im["file_name"] for im in d["images"]}
        for im in d["images"]:
            ts, offs, _ = parse(im["file_name"])
            clips[(split, ts, offs)].append(im["file_name"])
        for a in d["annotations"]:
            if a["category_id"] in (LITTER, GARBAGE):
                gt[split][id2fn[a["image_id"]]].append((a["category_id"], a["bbox"]))

    rows, summary = [], collections.Counter()
    for (split, ts, offs), names in sorted(clips.items()):
        key = f"{split}/{ts}_{offs//60:02d}m{offs%60:02d}s"
        ckey = f"{ts}_{offs//60:02d}m{offs%60:02d}s"
        info = align[key]
        outdir = V18 / f"{split}_changemap"
        outdir.mkdir(parents=True, exist_ok=True)

        # ── pass1：暖機 ──────────────────────────────────────────
        knn = cv2.createBackgroundSubtractorKNN(history=HIST, dist2Threshold=DIST, detectShadows=True)
        cap = cv2.VideoCapture(info["avi"])
        if key in TAIL_WARMUP:
            wr = range(TAIL_WARMUP[key]["start"], TAIL_WARMUP[key]["end"]); warm_kind = "尾巴C"
        else:
            wr = range(info["warm_start"], info["first_abs"]);              warm_kind = "pre-roll"
        n_warm = 0
        if len(wr) > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, wr.start)
            for _ in wr:
                ok, f = cap.read()
                if not ok:
                    break
                knn.apply(cv2.resize(f, None, fx=S, fy=S)); n_warm += 1
        cap.release()

        # ── pass2：凍結取前景 + sanity ───────────────────────────
        frs = []
        n_black = 0
        lfg, gfg = [], []
        for name in names:
            frs.append(int(re.search(r"_frame_(\d+)", name).group(1)))
            outp = outdir / (name[:-4] + ".png")
            img = cv2.imread(str(V18 / split / name))
            H, W = img.shape[:2]
            if outp.exists():
                cm = cv2.imread(str(outp), cv2.IMREAD_GRAYSCALE)
            elif n_warm == 0:
                cm = np.zeros((H, W), np.uint8); cv2.imwrite(str(outp), cm)
            else:
                fg = knn.apply(cv2.resize(img, None, fx=S, fy=S), learningRate=0)
                cm = cv2.resize(rm5(fg), (W, H), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(str(outp), cm)
            if cm.max() == 0:
                n_black += 1
            for cat, bbox in gt[split].get(name, []):
                r = box_fg_ratio(cm, bbox)
                if r is not None:
                    (lfg if cat == LITTER else gfg).append(r)
            summary["saved"] += 1

        note = info["note"]
        rows.append(dict(
            split=split, clip=ckey, wmin=min(frs), wmax=max(frs), n=len(names),
            manual=(ckey in MANUAL), fps=info["fps"], first_abs=info["first_abs"],
            diff=info["diff"], slope=info["slope"], warm_kind=warm_kind, n_warm=n_warm,
            lfg=(sum(lfg)/len(lfg) if lfg else None), ln=len(lfg),
            gfg=(sum(gfg)/len(gfg) if gfg else None), gn=len(gfg),
            n_black=n_black, note=note,
        ))
        summary["clip"] += 1
        if warm_kind == "尾巴C": summary["尾巴C"] += 1
        if info["preroll"] < WARM: summary["preroll<WARM"] += 1
        print(f"[{split:5s}] {ckey} 暖機{n_warm}幀({warm_kind}) 窗口{min(frs)}-{max(frs)}/{len(names)} "
              f"litter框fg={'%.2f'%(sum(lfg)/len(lfg)) if lfg else '-'}({len(lfg)}) 全黑{n_black}/{len(names)}")
        sys.stdout.flush()

    # ── 產出 MD ─────────────────────────────────────────────────
    def pct(x): return "-" if x is None else f"{x*100:.0f}%"
    with open(MD, "w", encoding="utf-8") as f:
        f.write("# 第18次 v18 changemap 逐 clip 處理記錄\n\n")
        f.write(f"方法：pre-roll 凍結背景兩段式 KNN(500/500/shadow)+rm5、half-res→INTER_NEAREST。"
                f"對齊表 alignment_v18.json。\n\n")
        f.write(f"- 總 clip {summary['clip']}、changemap {summary['saved']} 張\n")
        f.write(f"- 暖機：pre-roll {summary['clip']-summary['尾巴C']} 部、尾巴暖機(method C) {summary['尾巴C']} 部\n")
        f.write(f"- pre-roll<150（有多少餵多少）：{summary['preroll<WARM']} 部\n")
        f.write("- **框內前景比 fg%**：changemap 在 litter/garbage GT 框內的前景像素占比（越高＝越亮）。"
                "靜止落地物該低甚至全黑（BGS 設計如此）；動態物該亮。數字如實列，不下判讀。\n")
        f.write("- **全黑幀**：整張 changemap 全 0 的幀數。\n\n")
        f.write("| # | split | clip (ts_MMmSSs) | 窗口(起-訖/n) | 手動 | fps | first_abs(diff/斜率) | 暖機 | WARM幀 | "
                "litter框fg%(框數) | garbage框fg%(框數) | 全黑幀 | 備註 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for i, r in enumerate(sorted(rows, key=lambda r: (r["split"], r["clip"])), 1):
            f.write(f"| {i} | {r['split']} | {r['clip']} | {r['wmin']}-{r['wmax']}/{r['n']} | "
                    f"{'✔' if r['manual'] else ''} | {r['fps']:.0f} | {r['first_abs']}({r['diff']}/{r['slope']}) | "
                    f"{r['warm_kind']} | {r['n_warm']} | {pct(r['lfg'])}({r['ln']}) | {pct(r['gfg'])}({r['gn']}) | "
                    f"{r['n_black']}/{r['n']} | {r['note']} |\n")
    print(f"\n✅ 完成：{summary['clip']} clips，changemap {summary['saved']} 張。逐 clip 記錄 → {MD}")

if __name__ == "__main__":
    main()
