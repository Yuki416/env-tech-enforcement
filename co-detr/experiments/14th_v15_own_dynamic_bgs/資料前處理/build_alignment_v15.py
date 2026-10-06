#!/usr/bin/env python3
"""Step 0（第 14 次實驗 / v15）：建立每部 clip 的「首幀 → 原始 .avi 絕對索引」對齊表。

背景 / 為什麼需要這一步
----------------------
v15 的 changemap 用「事件前 pre-roll 凍結背景」的兩段式 KNN 生成（見 gen_changemaps_v15.py）。
第二段前景直接算在 dataset jpg 上，所以 .avi 唯一的用途是「提供 clip 開始之前的 WARM 幀當暖機素材」。
因此我們只需要定位「clip 的第一張要用的幀」落在 .avi 的哪個絕對索引，往前退 WARM 幀即可。

檔名慣例（v14 = RoboFlow 匯出）：
    Day_Garbage_..._<14位timestamp>_<MM>m<SS>s_RoboFlow_frame_<NNNN>.jpg
    - <MM>m<SS>s ：此 clip 在原始 .avi 中的起始時間偏移（有 ±1 秒截斷誤差）
    - frame_<NNNN>：clip *內部* 的幀編號（從 1 起算，非 .avi 絕對索引）

實測注意事項（已驗證，見對話記錄）：
    - .avi fps 有 10 與 12 兩種，且 metadata 不完全可信。
    - clip 幀與 .avi 幀 **非 1:1**（RoboFlow 以固定 10fps 重新取樣，斜率 1.2~1.44）。
      → 所以不能用單一常數把「任意」clip 幀映回 .avi。但**首幀對齊是穩的**（誤差 <2），
        而我們只需要首幀，尺度漂移與本步驟無關。
    - .avi 解析度 == dataset jpg 解析度（壓縮雜訊 <2），暖機(.avi)與查詢(jpg)可混用同一影像空間。

輸出：alignment_v15.json —— { "<split>/<ts>_<MMmSSs>": {avi, fps, first_abs, warm_start, preroll, diff, note} }
fail loud：任何 clip 首幀對不上（diff >= THRESH）或找不到 .avi 就 raise，不默默跳過。
"""
import json, re, glob, collections, sys
from pathlib import Path
import cv2, numpy as np

V14   = Path("/path/to/data/coco/chiayi_v14_own_dynamic_litter")
SANHE = "/path/to/data/raw_videos"
OUT   = Path(__file__).resolve().parent / "alignment_v15.json"
WARM  = 150
THRESH = 3.0          # 首幀平均絕對差門檻（實測全部 <2，留一點餘裕）
SEARCH = 6            # 首幀在預估位置 ± SEARCH 幀內搜尋
CHECK_THRESH = 8.0    # 抽樣校驗幀門檻（因尺度漂移，抽樣幀只要求粗略對齊，非嚴格）

def avi_index():
    m = {}
    for p in glob.glob(f"{SANHE}/**/*.avi", recursive=True):
        g = re.search(r"(\d{14})", p)
        if g:
            m.setdefault(g.group(1), p)
    return m

def parse(name):
    g = re.search(r"(\d{14})_(\d{2})m(\d{2})s", name)
    ts = g.group(1)
    offs = int(g.group(2)) * 60 + int(g.group(3))
    fn = int(re.search(r"_frame_(\d+)", name).group(1))
    return ts, offs, fn

def small(img):
    return cv2.resize(img, (320, 240)).astype(int)

def read_at(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    return f if ok else None

def main():
    avimap = avi_index()
    # key = (split, ts, offs) → list of (fn, filename)
    clips = collections.defaultdict(list)
    for split in ["train", "valid", "test"]:
        d = json.load(open(V14 / f"annotations/{split}.json"))
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
        ds0 = small(cv2.imread(str(V14 / split / name0)))

        # 首幀搜尋：mmss 可能 -1 秒截斷 → 試 d_sec ∈ {0,1}
        best = (1e9, None)
        for d_sec in (0, 1):
            base = int(round((offs + d_sec) * fps)) + fn0 - 1
            for idx in range(max(0, base - SEARCH), base + SEARCH + 1):
                f = read_at(cap, idx)
                if f is None:
                    continue
                diff = np.abs(small(f) - ds0).mean()
                if diff < best[0]:
                    best = (diff, idx)
        diff0, first_abs = best
        if first_abs is None or diff0 >= THRESH:
            cap.release()
            fails.append((split, ts, offs, f"首幀對不上 diff={diff0:.2f}"))
            continue

        # 校驗：用固定 K=first_abs-fn0 檢查另外幾張抽樣幀（僅供參考，尺度漂移下可能偏大）
        K = first_abs - fn0
        checks = []
        for fn, name in [lst[len(lst) // 3], lst[2 * len(lst) // 3], lst[-1]]:
            f = read_at(cap, K + fn)
            if f is None:
                checks.append(None)
            else:
                checks.append(round(float(np.abs(small(f) - small(cv2.imread(str(V14 / split / name)))).mean()), 2))
        cap.release()

        preroll = first_abs                      # .avi frame 0 → first_abs 之間可用的暖機幀數
        warm_start = max(0, first_abs - WARM)
        note = ""
        if preroll < 30:
            note = "pre-roll不足(<30)"
        elif preroll < WARM:
            note = f"pre-roll={preroll}<WARM，有多少餵多少"
        table[f"{split}/{ts}_{offs//60:02d}m{offs%60:02d}s"] = dict(
            split=split, ts=ts, offs=offs, avi=avi, fps=round(fps, 2),
            first_abs=first_abs, warm_start=warm_start, preroll=preroll,
            n_used=len(lst), diff=round(float(diff0), 3), checks=checks, note=note,
        )
        flag = f"  ⚠️ {note}" if note else ""
        print(f"[{split:5s}] {ts} {offs//60:02d}m{offs%60:02d}s fps={fps:.0f} "
              f"首幀idx={first_abs}(差{diff0:.2f}) pre-roll={preroll} 用幀={len(lst)}{flag}")
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
