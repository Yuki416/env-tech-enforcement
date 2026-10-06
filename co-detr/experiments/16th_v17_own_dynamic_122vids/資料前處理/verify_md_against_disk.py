#!/usr/bin/env python3
"""獨立驗證：動態區間對照 Markdown 裡的數字，是否與『實際落盤資料集』一致。

刻意**不重用** build/analyze 的選幀邏輯，只做兩件事：
  1. 直接讀三個實際落盤資料集的 JSON，逐 clip 算真實 frame (min, max, count)：
     - HEAD=15  : chiayi_v17_own_dynamic_litter
     - HEAD=10  : chiayi_v17_own_dynamic_litter_head10
     - 第15次    : chiayi_v16_own_dynamic_litter
  2. 解析 MD 表格每一列的「HEAD=10 取到 / HEAD=15 取到 / 第15次實際」三欄，
     與上面的 ground truth 逐格比對。

另外做兩個一致性檢查（把「種子」與落盤綁在一起）：
  A. 每個 clip：落盤 HEAD=15 與 HEAD=10 的**後緣相同**（種子/尾巴不隨 HEAD 變）。
  B. 每個 clip：落盤 HEAD=15 前緣 − HEAD=10 前緣 == 5，除非其中一個已夾到片頭(=最小可用幀)。
     （HEAD 差 5，前緣就該差 5，除非撞到片頭被 clamp。）

全部通過才印 ✅。用法：PYTHONUTF8=1 python3 verify_md_against_disk.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MD = HERE / "動態區間對照_HEAD10_vs_HEAD15_vs_15th.md"
DS = {
    "h15": Path("/path/to/data/coco/chiayi_v17_own_dynamic_litter"),
    "h10": Path("/path/to/data/coco/chiayi_v17_own_dynamic_litter_head10"),
    "v15": Path("/path/to/data/coco/chiayi_v16_own_dynamic_litter"),
}
frame_re = re.compile(r"^(.*)_frame_(\d+)\.jpg$", re.IGNORECASE)
KEY_RE = re.compile(r"_(\d{14})_((?:\d{2}h)?\d{2}m\d{2}s)")


def ts_key(clip):
    m = KEY_RE.search(clip)
    return (m.group(1), m.group(2)) if m else None


def short(clip):
    """與 analyze 腳本的 short() 完全相同的縮寫規則。"""
    k = ts_key(clip)
    tag = "_".join(clip.split("_")[0:2])
    return f"{tag}…{k[0][-6:]}_{k[1]}" if k else clip[:32]


def disk_ranges(ds_path, split):
    """{clip: (min,max,count)}，逐 clip 由磁碟 JSON 直接算。"""
    j = json.load(open(ds_path / "annotations" / f"{split}.json", encoding="utf-8"))
    by = defaultdict(list)
    for im in j["images"]:
        m = frame_re.match(im["file_name"])
        if m:
            by[m.group(1)].append(int(m.group(2)))
    return {c: (min(v), max(v), len(v)) for c, v in by.items()}


def clip_all_frames(ds_path, split):
    """{clip: sorted[所有幀]}，用於判斷片頭最小可用幀（來源=完整 v17）。"""
    j = json.load(open(ds_path / "annotations" / f"{split}.json", encoding="utf-8"))
    by = defaultdict(list)
    for im in j["images"]:
        m = frame_re.match(im["file_name"])
        if m:
            by[m.group(1)].append(int(m.group(2)))
    return {c: sorted(v) for c, v in by.items()}


def parse_cell(cell):
    """'1–17 (17)' -> (1,17,17); '1–13' -> (1,13,None); '—'/'無此clip' -> None"""
    cell = cell.strip()
    if "無此" in cell or cell == "—":
        return None
    m = re.search(r"(\d+)[–\-](\d+)(?:\s*\((\d+)\))?", cell)
    if not m:
        return "PARSE_FAIL"
    return (int(m.group(1)), int(m.group(2)),
            int(m.group(3)) if m.group(3) else None)


def parse_md():
    """回傳 [{'clip_short','h10','h15','v15'}]，只取資料列。"""
    rows = []
    cur_split = None
    for ln in MD.read_text(encoding="utf-8").splitlines():
        if ln.startswith("### train"):
            cur_split = "train"
        elif ln.startswith("### valid"):
            cur_split = "valid"
        if not ln.startswith("|"):
            continue
        cols = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cols) < 6:
            continue
        if cols[0] in ("clip",) or set(cols[0]) <= set("-: "):
            continue  # 表頭/分隔線
        clip_cell = re.sub(r"\s*\*\(.*?\)\*", "", cols[0]).strip()  # 去掉 *(...)* 標記
        rows.append({
            "split": cur_split, "clip_short": clip_cell,
            "seeds": cols[1], "h10": parse_cell(cols[2]),
            "h15": parse_cell(cols[3]), "v15": parse_cell(cols[4]),
        })
    return rows


def main():
    # ground truth（由磁碟直接算），key = short 名
    gt = {}
    for tag, path in DS.items():
        gt[tag] = {}
        for split in ["train", "valid"]:
            for clip, rng in disk_ranges(path, split).items():
                gt[tag][(split, short(clip))] = rng
    # 完整 v17 每 clip 的最小可用幀（判斷 clamp）
    src = Path("/path/to/data/coco/chiayi_v17_garbage_litter_refined_122vids")
    minframe = {}
    for split in ["train", "valid"]:
        for clip, fr in clip_all_frames(src, split).items():
            minframe[(split, short(clip))] = fr[0]

    rows = parse_md()
    print(f"MD 解析到 {len(rows)} 個資料列\n")

    mism = []
    # ---- 逐格比對 MD vs 磁碟 ground truth ----
    for r in rows:
        keysplit = (r["split"], r["clip_short"])
        for tag in ["h10", "h15", "v15"]:
            md_val = r[tag]
            disk_val = gt[tag].get(keysplit)
            if md_val == "PARSE_FAIL":
                mism.append((keysplit, tag, "MD 解析失敗", r))
                continue
            if md_val is None:
                # MD 說「無此 clip」→ 磁碟也應無
                if disk_val is not None:
                    mism.append((keysplit, tag, f"MD=無 但磁碟有 {disk_val}", None))
                continue
            if disk_val is None:
                mism.append((keysplit, tag, f"MD={md_val} 但磁碟無此 clip", None))
                continue
            # 比 min,max；count 若 MD 有寫也要比
            md_mn, md_mx, md_cnt = md_val
            dk_mn, dk_mx, dk_cnt = disk_val
            if (md_mn, md_mx) != (dk_mn, dk_mx) or (md_cnt is not None and md_cnt != dk_cnt):
                mism.append((keysplit, tag, f"MD={md_val} vs 磁碟={disk_val}", None))

    # ---- 一致性檢查 A/B（用磁碟 h10/h15）----
    consist = []
    for keysplit in gt["h15"]:
        r15 = gt["h15"].get(keysplit)
        r10 = gt["h10"].get(keysplit)
        if not r15 or not r10:
            continue
        # A. 後緣相同
        if r15[1] != r10[1]:
            consist.append((keysplit, f"後緣不同 h15={r15[1]} h10={r10[1]}"))
        # B. HEAD=10 前緣應比 HEAD=15 前緣晚 5 幀（窗更小），除非撞片頭被 clamp
        mf = minframe.get(keysplit, 1)
        if (r10[0] - r15[0]) != 5:
            # 允許：兩者其一已到片頭最小可用幀（被 clamp）
            if not (r10[0] == mf or r15[0] == mf):
                consist.append((keysplit, f"前緣差≠5 h15前緣={r15[0]} h10前緣={r10[0]} 片頭={mf}"))

    # ---- 報告 ----
    print("=" * 70)
    if mism:
        print(f"❌ MD vs 磁碟：{len(mism)} 處不一致")
        for ks, tag, msg, _ in mism[:30]:
            print(f"   [{ks[0]}] {ks[1]}  {tag}: {msg}")
    else:
        print(f"✅ MD vs 磁碟：全部一致（h10/h15/第15次 三欄、min/max/count 逐格相符）")
    print("-" * 70)
    if consist:
        print(f"⚠️ 種子一致性檢查：{len(consist)} 處異常")
        for ks, msg in consist[:30]:
            print(f"   [{ks[0]}] {ks[1]}  {msg}")
    else:
        print("✅ 種子一致性：所有 clip 落盤 HEAD15/HEAD10 後緣相同、前緣差恰為5(或撞片頭)")
    print("=" * 70)

    if mism or consist:
        sys.exit(1)
    print("\n🎉 全部驗證通過：MD 數字 == 實際落盤資料集。")


if __name__ == "__main__":
    main()
