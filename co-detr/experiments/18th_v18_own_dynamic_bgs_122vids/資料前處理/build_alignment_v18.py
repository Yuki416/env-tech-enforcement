#!/usr/bin/env python3
"""第18次 / v18：建立每部 clip 的「最早使用幀 → 原始 .avi 絕對索引」對齊表。
（沿用第14次 build_alignment_v15.py，針對 v18 資料集加兩個修正。）

用途：changemap 用「事件前 pre-roll 凍結背景」兩段式 KNN（見 gen_changemaps_v18.py）。
pass2 前景直接算在 dataset jpg 上，.avi 只用來提供「clip 開始前的 WARM 幀」暖機。
故只需定位「每部 clip 最早要用的那張幀」落在 .avi 的絕對索引 first_abs，往前退 WARM 幀當 pre-roll。

相對 v15 的兩個修正（v18 才會踩到，已在真資料驗證）：
  ① 檔名偏移支援「小時」：clip 20240824171830 的偏移是 01h12m28s（=4348s）。
     v15 的 parse 只認 <MM>m<SS>s → 該 clip 整個漏掉。改成 (?:(HH)h)?(MM)m(SS)s。
  ② 多斜率首幀搜尋：v18 動態窗口起點常 ≫1（最高 fn0=34）。v15 用 base=offs*fps+(fn0-1)（斜率當1），
     在 fn0 大、真實斜率 1.2~1.44 時漂移可達 ~15 幀 > SEARCH=6 → 對不上。
     改成對 SLOPES∈{1.0..1.5} 各自 base=round((offs+d)*fps)+round((fn0-1)*slope)、各搜 ±SEARCH，取全域最小差。

輸出：alignment_v18.json —— { "<split>/<ts>_<MMmSSs>": {avi, fps, first_abs, warm_start, preroll, slope, diff, checks, note} }
fail loud：任何 clip 首幀對不上（diff>=THRESH）或找不到 .avi 就 raise，不默默跳過。
"""
import json, re, glob, collections, sys
from pathlib import Path
import cv2, numpy as np

V18   = Path("/path/to/data/coco/chiayi_v18_own_dynamic_bgs_122vids")
SANHE = "/path/to/data/raw_videos"
OUT   = Path(__file__).resolve().parent / "alignment_v18.json"
WARM   = 150
THRESH = 3.0          # 首幀平均絕對差門檻（實測全部 <1.6，留餘裕）
SEARCH = 6            # 每個 (d_sec, slope) base 附近 ± 搜尋幀數
SLOPES = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]   # ② clip↔.avi 重取樣斜率候選

def avi_index():
    m = {}
    for p in glob.glob(f"{SANHE}/**/*.avi", recursive=True):
        g = re.search(r"(\d{14})", p)
        if g:
            m.setdefault(g.group(1), p)
    return m

def parse(name):
    # ① 支援可選小時：<14ts>_(HHh)?MMmSSs_..._frame_NNNN
    g = re.search(r"(\d{14})_(?:(\d{2})h)?(\d{2})m(\d{2})s", name)
    h = int(g.group(2)) if g.group(2) else 0
    offs = h * 3600 + int(g.group(3)) * 60 + int(g.group(4))
    fn = int(re.search(r"_frame_(\d+)", name).group(1))
    return g.group(1), offs, fn

def small(img):
    return cv2.resize(img, (320, 240)).astype(int)

def read_at(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    return f if ok else None

def main():
    avimap = avi_index()
    clips = collections.defaultdict(list)
    for split in ["train", "valid", "test"]:
        d = json.load(open(V18 / f"annotations/{split}.json"))
        for im in d["images"]:
            ts, offs, fn = parse(im["file_name"])
            clips[(split, ts, offs)].append((fn, im["file_name"]))

    print(f"共 {len(clips)} 部 clip")
    table, fails = {}, []
    for (split, ts, offs), lst in sorted(clips.items()):
        lst.sort()
        if ts not in avimap:
            fails.append((split, ts, offs, "無對應 .avi"))
            continue
        avi = avimap[ts]
        cap = cv2.VideoCapture(avi)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fn0, name0 = lst[0]
        ds0 = small(cv2.imread(str(V18 / split / name0)))

        # ② 首幀搜尋：offs 可能 -1 秒截斷 → d_sec∈{0,1}；斜率未知 → 多斜率候選，各搜 ±SEARCH
        best = (1e9, None, None)   # (diff, idx, slope)
        for d_sec in (0, 1):
            for sl in SLOPES:
                base = int(round((offs + d_sec) * fps)) + int(round((fn0 - 1) * sl))
                for idx in range(max(0, base - SEARCH), base + SEARCH + 1):
                    f = read_at(cap, idx)
                    if f is None:
                        continue
                    diff = np.abs(small(f) - ds0).mean()
                    if diff < best[0]:
                        best = (diff, idx, sl)
        diff0, first_abs, slope = best
        if first_abs is None or diff0 >= THRESH:
            cap.release()
            fails.append((split, ts, offs, f"首幀對不上 diff={diff0:.2f}"))
            continue

        # 校驗：用命中的 slope 外推抽樣幀（僅供參考，尺度漂移下仍可能偏大）
        checks = []
        for fn, name in [lst[len(lst) // 3], lst[2 * len(lst) // 3], lst[-1]]:
            idx = first_abs + int(round((fn - fn0) * slope))
            f = read_at(cap, idx)
            checks.append(None if f is None else
                          round(float(np.abs(small(f) - small(cv2.imread(str(V18 / split / name)))).mean()), 2))
        cap.release()

        preroll = first_abs
        warm_start = max(0, first_abs - WARM)
        note = ""
        if preroll == 0:
            note = "pre-roll=0（需尾巴暖機 method C）"
        elif preroll < 30:
            note = "pre-roll不足(<30)"
        elif preroll < WARM:
            note = f"pre-roll={preroll}<WARM，有多少餵多少"
        table[f"{split}/{ts}_{offs//60:02d}m{offs%60:02d}s"] = dict(
            split=split, ts=ts, offs=offs, avi=avi, fps=round(fps, 2),
            first_abs=first_abs, warm_start=warm_start, preroll=preroll,
            slope=slope, n_used=len(lst), diff=round(float(diff0), 3),
            checks=checks, note=note,
        )
        flag = f"  ⚠️ {note}" if note else ""
        print(f"[{split:5s}] {ts} {offs//60:02d}m{offs%60:02d}s fps={fps:.0f} "
              f"起點fn0={fn0} 首幀idx={first_abs}(差{diff0:.2f},斜率{slope}) "
              f"pre-roll={preroll} 用幀={len(lst)}{flag}")
        sys.stdout.flush()

    if fails:
        print("\n❌ 以下 clip 無法對齊（中止，不寫出）:")
        for split, ts, offs, why in fails:
            print(f"   [{split}] {ts} {offs//60:02d}m{offs%60:02d}s — {why}")
        raise SystemExit(f"有 {len(fails)} 部 clip 對齊失敗，請人工檢查後再繼續。")

    json.dump(table, open(OUT, "w"), ensure_ascii=False, indent=2)
    b = collections.Counter()
    for v in table.values():
        p = v["preroll"]
        b["≥150" if p >= 150 else "100-149" if p >= 100 else "30-99" if p >= 30 else "<30"] += 1
    print(f"\n✅ 全 {len(table)} 部對齊成功，寫出 {OUT}")
    print("pre-roll 分佈:", dict(b))
    print("pre-roll <WARM 的 clip:")
    for k, v in sorted(table.items(), key=lambda kv: kv[1]["preroll"]):
        if v["preroll"] < WARM:
            print(f"   {k}  preroll={v['preroll']}  用幀={v['n_used']}  {v['note']}")

if __name__ == "__main__":
    main()
