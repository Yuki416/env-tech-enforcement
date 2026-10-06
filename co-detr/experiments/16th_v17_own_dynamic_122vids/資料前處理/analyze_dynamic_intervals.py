#!/usr/bin/env python3
"""分析並輸出第 16 次（v17）各 clip 的動態擷取區間，比較 HEAD=10 / 15，並對照第 15 次（HEAD=30）。

回答三個問題（使用者要求）：
  1. HEAD=10 與 HEAD=15 各自取到的「動態區間」為何？
  2. 種子區間前後再加上 HEAD/TAIL padding 後，實際取到的區間變成什麼？
  3. 這一次取的區間，跟第 15 次（v16、HEAD=30）有沒有不同？

機制重點：
  - 「種子」= litter/garbage 相鄰幀位移 ≥ SEED_THRESH 的幀（外加人工 MANUAL_DYNAMIC 種子）。
    種子**不受 HEAD 影響**，HEAD=10/15/30 的種子完全相同。
  - 「取到的區間」= [首種子 − HEAD, 末種子 + TAIL]（套 END_CAP、夾在該 clip 可用幀內）。
    → HEAD 只影響**前緣**；TAIL 不變 → 後緣不變（除非 END_CAP）。
  - 例外：本次重標的 7 部，標註改了 → 連「種子」都可能和第 15 次不同（不只前緣）。

對照第 15 次：直接讀第 15 次落盤資料集 chiayi_v16_own_dynamic_litter 每個 clip 的實際幀範圍，
以 (時間戳, 片長) 為 join key（跨 _RoboFlow 改名仍可對上）。

輸出：動態區間對照_HEAD10_vs_HEAD15_vs_15th.md（本資料夾）
用法：PYTHONUTF8=1 python3 analyze_dynamic_intervals.py
"""
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_MD = HERE / "動態區間對照_HEAD10_vs_HEAD15_vs_15th.md"
V15_DYN = Path("/path/to/data/coco/chiayi_v16_own_dynamic_litter")  # 第15次落盤
MANUAL_DS = Path("/path/to/data/coco/chiayi_v17_own_dynamic_litter_head10_manual")  # 手動微調版(基底HEAD10)

# ---- 匯入 build_v17_dynamic_litter 以重用選幀邏輯 ----
spec = importlib.util.spec_from_file_location("build_v17", HERE / "build_v17_dynamic_litter.py")
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)

KEY_RE = re.compile(r"_(\d{14})_((?:\d{2}h)?\d{2}m\d{2}s)")
frame_re = re.compile(r"^(.*)_frame_(\d+)\.jpg$", re.IGNORECASE)


def ts_key(clip):
    """由 clip 名抽 (時間戳, 片長) 當跨版本 join key。"""
    m = KEY_RE.search(clip)
    return (m.group(1), m.group(2)) if m else None


def effective_seeds_per_clip(imgs, seed_frames):
    """auto 種子（seed_frames）依 clip 分組的幀號；再加 MANUAL 種子端點。"""
    auto = defaultdict(list)
    for iid in seed_frames:
        auto[imgs[iid]["clip"]].append(imgs[iid]["frame"])
    eff = {c: sorted(v) for c, v in auto.items()}
    # 併入人工種子（與 build 相同：MANUAL_DYNAMIC / MANUAL_DYNAMIC_V16_NEW）
    all_clips = {imgs[i]["clip"] for i in imgs}
    for clip in all_clips:
        for table in (B.MANUAL_DYNAMIC, B.MANUAL_DYNAMIC_V16_NEW):
            hit = B.lookup(clip, table)
            if hit:
                eff.setdefault(clip, [])
                eff[clip].extend([hit[0], hit[1]])
                eff[clip] = sorted(set(eff[clip]))
    return eff


def v17_intervals(split, head):
    """回傳 {clip: {'seeds':(mn,mx)|None, 'sel':(mn,mx,count)|None, 'manual':bool}}（v17，指定 HEAD）。"""
    B.HEAD_WINDOW = head
    j, imgs, seed_frames, selected, seeds_per_cat, all_clips, auto_seeded = B.select_frames(split)
    eff = effective_seeds_per_clip(imgs, seed_frames)
    sel_by_clip = defaultdict(list)
    for iid in selected:
        sel_by_clip[imgs[iid]["clip"]].append(imgs[iid]["frame"])
    out = {}
    for clip in all_clips:
        s = eff.get(clip)
        sel = sorted(sel_by_clip.get(clip, []))
        is_manual = bool(B.lookup(clip, B.MANUAL_DYNAMIC) or B.lookup(clip, B.MANUAL_DYNAMIC_V16_NEW))
        out[clip] = {
            "seeds": (min(s), max(s)) if s else None,
            "sel": (sel[0], sel[-1], len(sel)) if sel else None,
            "manual": is_manual,
        }
    return out


def v15_intervals(split):
    """第 15 次落盤：{ (ts,dur): (clip, mn, mx, count) }。"""
    ann = V15_DYN / "annotations" / f"{split}.json"
    j = json.load(open(ann, encoding="utf-8"))
    by_clip = defaultdict(list)
    for im in j["images"]:
        m = frame_re.match(im["file_name"])
        if m:
            by_clip[m.group(1)].append(int(m.group(2)))
    out = {}
    for clip, frames in by_clip.items():
        k = ts_key(clip)
        if k:
            out[k] = (clip, min(frames), max(frames), len(frames))
    return out


def fmt_range(t):
    if t is None:
        return "—"
    if len(t) == 2:
        return f"{t[0]}–{t[1]}"
    return f"{t[0]}–{t[1]} ({t[2]})"


def short(clip):
    """縮短 clip 名以利表格閱讀：保留 類別_攝影機_地點省略_時間戳末段_片長。"""
    k = ts_key(clip)
    prefix = clip.split("_")[0:2]  # e.g. Day, Litter
    tag = "_".join(prefix)
    if k:
        return f"{tag}…{k[0][-6:]}_{k[1]}"
    return clip[:32]


def manual_ranges(split):
    """手動微調版落盤：{ (ts,dur): (clip, mn, mx, count) }；資料夾不存在則回空。"""
    ann = MANUAL_DS / "annotations" / f"{split}.json"
    if not ann.exists():
        return {}
    j = json.load(open(ann, encoding="utf-8"))
    by_clip = defaultdict(list)
    for im in j["images"]:
        m = frame_re.match(im["file_name"])
        if m:
            by_clip[m.group(1)].append(int(m.group(2)))
    out = {}
    for clip, frames in by_clip.items():
        k = ts_key(clip)
        if k:
            out[k] = (clip, min(frames), max(frames), len(frames))
    return out


def build_split_section(split, v17_h10, v17_h15, v15, stats):
    reannot_keys = {ts_key(c) for c in B_REANNOT}
    lines = []
    lines.append(f"### {split}（共 {len(v17_h15)} clips）\n")
    lines.append("| clip | 種子區間 | **HEAD=10 取到** | **HEAD=15 取到** | 第15次(HEAD=30)實際 | 與第15次差異 |")
    lines.append("|---|---|---|---|---|---|")
    # 排序：重標 7 部優先，其餘依名稱
    def sort_key(c):
        k = ts_key(c)
        return (0 if k in reannot_keys else 2, c)
    for clip in sorted(v17_h15, key=sort_key):
        k = ts_key(clip)
        d10, d15 = v17_h10[clip], v17_h15[clip]
        v15row = v15.get(k)
        seeds = fmt_range(d10["seeds"])  # 種子與 HEAD 無關，取 h10 即可
        sel10 = fmt_range(d10["sel"])
        sel15 = fmt_range(d15["sel"])
        v15sel = fmt_range((v15row[1], v15row[2], v15row[3])) if v15row else "（第15次無此clip）"
        # 差異判定
        note = []
        is_re = k in reannot_keys
        if is_re:
            note.append("🔧重標")
        if v15row is None:
            note.append("🆕新增")
            stats["new"] += 1
        else:
            # 後緣不受 HEAD 影響 → 後緣不同代表種子/標註相對第15次有變
            if d15["sel"] and v15row and d15["sel"][1] != v15row[2]:
                note.append(f"後緣 {v15row[2]}→{d15['sel'][1]}")
                stats["backedge"] += 1
        if is_re:
            stats["reannot"] += 1
        elif v15row is not None:
            stats["front_only"] += 1
        tag = "，".join(note) if note else "僅前緣隨HEAD縮"
        # 標記：人工種子 / END_CAP / TAIL_OVERRIDE
        marks = []
        if d15["manual"]:
            marks.append("人工種子")
        if B.lookup(clip, B.END_CAP) is not None:
            marks.append(f"END_CAP={B.lookup(clip, B.END_CAP)}")
        if B.lookup(clip, B.TAIL_OVERRIDE) is not None:
            marks.append(f"TAIL={B.lookup(clip, B.TAIL_OVERRIDE)}")
        cm = f" *({'／'.join(marks)})*" if marks else ""
        lines.append(f"| {short(clip)}{cm} | {seeds} | {sel10} | {sel15} | {v15sel} | {tag} |")
    lines.append("")
    return "\n".join(lines)


# 相對第 15 次(v16) 被重標過的 clip，種子可能與 15th 不同（非 HEAD 效果）。
# = 首建重標 7 部 + R2 修正輪 4 部 = 11 部（以 (時間戳,片長) 比對，見 ts_key）。
B_REANNOT = [
    # 首建重標 7 部（train）
    "Day_Litter_12_8_loc12_bureau_20240603181958_04m49s",
    "Day_Litter_535_8_loc535_20240712183651_02m34s",
    "Day_Litter_536_8_loc536_20240721183149_07m22s",
    "Day_Litter_536_8_loc536_20240804064121_00m36s",
    "Day_Litter_536_8_loc536_20240807062933_02m17s",
    "Day_Litter_536_8_loc536_20240809180115_04m55s",
    "Day_Litter_543_8_loc543_20240610061309_00m27s",
    # R2 修正輪 4 部（3 train + 1 valid，2026-07-25）
    "Day_Litter_537_8_loc537_20240902175511_01m08s",
    "Day_Litter_538_8_loc538_20240624173521_06m19s",
    "Day_Litter_538_8_loc538_20240809160023_00m25s",
    "Day_Litter_537_8_loc537_20240908065812_02m13s",
]


def main():
    from collections import Counter
    sections = []
    totals = {}
    stats = Counter()
    overrides = []          # 手動覆寫的 clip：(split, short, 純h10範圍, 手動範圍)
    manual_present = False
    for split in ["train", "valid"]:
        h10 = v17_intervals(split, 10)
        h15 = v17_intervals(split, 15)
        v15 = v15_intervals(split)
        man = manual_ranges(split)
        sections.append(build_split_section(split, h10, h15, v15, stats))
        totals[split] = {
            "h10": sum(d["sel"][2] for d in h10.values() if d["sel"]),
            "h15": sum(d["sel"][2] for d in h15.values() if d["sel"]),
            "v15": sum(r[3] for r in v15.values()),
            "man": sum(r[3] for r in man.values()) if man else 0,
        }
        # 找手動版與純 HEAD=10 不同的 clip
        if man:
            manual_present = True
            for clip, d in h10.items():
                k = ts_key(clip)
                mr = man.get(k)
                if not d["sel"] or not mr:
                    continue
                pure = (d["sel"][0], d["sel"][1], d["sel"][2])
                cur = (mr[1], mr[2], mr[3])
                if pure != cur:
                    overrides.append((split, short(clip), pure, cur))

    # ---- 手動覆寫專區（只有手動版存在且有差異時才印）----
    if manual_present and overrides:
        ov_lines = ["", "## 手動覆寫（RANGE_OVERRIDE，手動 head10 版）", "",
                    f"> 手動版落盤：`{MANUAL_DS.name}`（基底 HEAD=10 + 逐 clip 直接指定最終頭尾幀，所見即所得、與 HEAD 無關）。",
                    "> 下列 clip 的最終擷取範圍由人工釘死，**取代**純 HEAD=10 的自動結果；其餘 clip 與純 HEAD=10 相同。",
                    "",
                    "| split | clip | 純 HEAD=10 | **手動(最終)** | 幀數變化 |",
                    "|---|---|---|---|---|"]
        for split, sh, pure, cur in overrides:
            ov_lines.append(f"| {split} | {sh} | {pure[0]}–{pure[1]} ({pure[2]}) | "
                            f"**{cur[0]}–{cur[1]} ({cur[2]})** | {pure[2]}→{cur[2]} |")
        ov_lines.append("")
        manual_section = "\n".join(ov_lines)
    else:
        manual_section = ""

    # ---- 總量對照表（手動版存在時多一欄）----
    mh = " **手動head10** |" if manual_present else ""
    ms = "---|" if manual_present else ""
    def mc(sp):
        return f" {totals[sp]['man']} 幀 |" if manual_present else ""
    totals_table = (
        f"| split | HEAD=10 | HEAD=15 | 第15次(HEAD=30) |{mh}\n"
        f"|---|---|---|---|{ms}\n"
        f"| train | {totals['train']['h10']} 幀 | {totals['train']['h15']} 幀 | {totals['train']['v15']} 幀 |{mc('train')}\n"
        f"| valid | {totals['valid']['h10']} 幀 | {totals['valid']['h15']} 幀 | {totals['valid']['v15']} 幀 |{mc('valid')}"
    )

    header = f"""# 動態擷取區間對照：HEAD=10 vs HEAD=15 vs 第15次(HEAD=30)

> 由 `analyze_dynamic_intervals.py` 自動產生。資料集 v17 動態子集，配方除 HEAD_WINDOW 外沿用第 15 次。
> SEED_THRESH={B.SEED_THRESH}px/幀、TAIL_WINDOW={B.TAIL_WINDOW}、GAP_CAP={B.GAP_CAP}。

## 結論：這次取的動態區間跟第 15 次差在哪

- **{stats['front_only']} 個 clip（未重標的 train+valid）：種子與第 15 次完全相同**，區間差異**純粹**來自
  HEAD 30→15/10 讓**前緣往後縮**（後緣不變）。這是本次唯一刻意動的變因。
- **{stats['reannot']} 個重標 clip**（首建 7 部 + R2 修正輪 4 部）：標註改了 → **連種子都可能和第 15 次不同**；
  其中 **{stats['backedge']} 個連後緣也變了**（見「後緣 A→B」標記，末種子位置相對第 15 次移動，非 HEAD 效果）。
- **{stats['new']} 個新增 clip**：第 15 次沒有（新增到 valid 的那部 garbage `…064620_00m19s`）。
- HEAD=10 vs HEAD=15 本身：**種子完全相同**，只差前緣多縮 5 幀（HEAD=10 更靠近丟棄動作）。

## 怎麼看這份表

- **種子區間** = litter/garbage 相鄰幀位移 ≥ {B.SEED_THRESH}px 的幀範圍（＋人工 MANUAL 種子）。
  **種子不受 HEAD 影響** → HEAD=10/15/30 的種子完全相同。
- **取到的區間** = `[首種子 − HEAD, 末種子 + TAIL={B.TAIL_WINDOW}]`，套 END_CAP、夾在該 clip 可用幀內。
  括號內數字 = 該 clip 選中的幀數。
- **HEAD 只影響前緣**：HEAD 30→15→10，前緣往內縮、後緣不變（除非 END_CAP）。
- 「與第15次差異」欄：
  - `僅前緣隨HEAD縮` = 種子與第15次相同，只是 HEAD 變小使前緣往後移（共 {stats['front_only']} 部，見上方結論）。
  - `🔧重標` = 相對第15次被重標過（首建 7 部 + R2 修正輪 4 部 = {stats['reannot']} 部），標註改了 →
    **連種子都可能和第15次不同**（不只前緣）。
  - `🆕新增` = 第15次沒有的 clip（{stats['new']} 部，新增到 valid 的那部 garbage）。
  - `後緣 A→B` = 後緣改變（代表種子/標註相對第15次有變，非單純 HEAD 效果）。

## 總量對照

{totals_table}
{manual_section}
---

## 逐 clip 明細
"""
    OUT_MD.write_text(header + "\n" + "\n".join(sections), encoding="utf-8")
    print(f"✅ 已輸出 {OUT_MD}")
    for sp, t in totals.items():
        print(f"  {sp}: HEAD10={t['h10']} HEAD15={t['h15']} 15th={t['v15']}")


if __name__ == "__main__":
    main()
