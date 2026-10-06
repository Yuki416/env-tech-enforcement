#!/usr/bin/env python3
"""第 16 次訓練正式資料集（手動微調版）最終驗證。

驗證 `chiayi_v17_own_dynamic_litter_head10_manual`：
  1. 三 split 磁碟↔JSON 一致、無重複 image_id/annotation_id/file_name、印規模與類別分布。
  2. 主判定 test_static594 = 594 張 / 1823 框。
  3. 8 部手動覆寫的實際幀範圍 == 預期。
  4. 非覆寫 clip 的檔名集合與「純 HEAD=10」完全相同（確認只有 8 部被動到）。
  5. 跨 split 檔名零重疊。
全綠才印「可正式採用」。用法：PYTHONUTF8=1 python3 final_check_manual.py
"""
import json, re, sys
from collections import defaultdict, Counter
from pathlib import Path

FINAL = Path("/path/to/data/coco/chiayi_v17_own_dynamic_litter_head10_manual")
PURE = Path("/path/to/data/coco/chiayi_v17_own_dynamic_litter_head10")
frame_re = re.compile(r"^(.*)_frame_(\d+)\.jpg$")
KEY = re.compile(r"_(\d{14})_((?:\d{2}h)?\d{2}m\d{2}s)")
CATS = {1: "car", 2: "litter", 3: "motorcycle", 4: "person", 5: "garbage"}
# 8 部手動覆寫 (時間戳,片長) -> 預期最終範圍（與 build_v17_manual_head10.py 的 RANGE_OVERRIDE 對應）
OVERRIDES = {
    ("20240828064620", "00m12s"): (6, 24), ("20240603181958", "04m49s"): (1, 22),
    ("20240826063130", "00m27s"): (1, 41), ("20240906054955", "01m00s"): (1, 32),
    ("20240802173010", "00m54s"): (12, 44), ("20240804171613", "24m02s"): (1, 28),
    ("20240911175753", "04m39s"): (6, 30), ("20240803182230", "06m22s"): (2, 22),
}


def tk(c):
    m = KEY.search(c)
    return (m.group(1), m.group(2)) if m else None


def main():
    allok = True
    print("=" * 60, "\n第 16 次訓練正式資料集 最終驗證\n", "=" * 60, sep="")
    for s in ["train", "valid", "test"]:
        j = json.load(open(FINAL / "annotations" / f"{s}.json", encoding="utf-8"))
        want = {i["file_name"] for i in j["images"]}
        have = {p.name for p in (FINAL / s).iterdir()}
        clips = {frame_re.match(i["file_name"]).group(1) for i in j["images"] if frame_re.match(i["file_name"])}
        cc = Counter(a["category_id"] for a in j["annotations"])
        dup_img = len(j["images"]) - len({i["id"] for i in j["images"]})
        dup_ann = len(j["annotations"]) - len({a["id"] for a in j["annotations"]})
        dup_fn = len(j["images"]) - len(want)
        ok = (want == have) and not (dup_img or dup_ann or dup_fn)
        allok &= ok
        print(f"\n[{s}] {'✅' if ok else '❌'} 影像{len(j['images'])} clips{len(clips)} 標註{len(j['annotations'])}"
              f"  重複 img/ann/fn={dup_img}/{dup_ann}/{dup_fn}")
        print("     " + " ".join(f"{CATS[k]}={cc[k]}" for k in sorted(CATS)))

    j = json.load(open(FINAL / "annotations/test_static594.json", encoding="utf-8"))
    ok = len(j["images"]) == 594 and len(j["annotations"]) == 1823
    allok &= ok
    print(f"\n[主判定] test_static594: {len(j['images'])}/{len(j['annotations'])} {'✅' if ok else '❌'}")

    print("\n[8 部手動覆寫 範圍]")
    for s in ["train", "valid"]:
        j = json.load(open(FINAL / "annotations" / f"{s}.json", encoding="utf-8"))
        byk = defaultdict(list)
        for i in j["images"]:
            m = frame_re.match(i["file_name"])
            if m:
                byk[tk(m.group(1))].append(int(m.group(2)))
        for k, (lo, hi) in OVERRIDES.items():
            if k in byk:
                fr = sorted(byk[k]); got = (fr[0], fr[-1]); ok = got == (lo, hi); allok &= ok
                print(f"   {'✅' if ok else '❌'} {k[0][-6:]}_{k[1]}: {got[0]}–{got[1]} (期望 {lo}–{hi})")

    print("\n[非覆寫 clip == 純 head10]")
    ovk = set(OVERRIDES)
    isov = lambda fn: (frame_re.match(fn) and tk(frame_re.match(fn).group(1)) in ovk)
    for s in ["train", "valid"]:
        ff = {i["file_name"] for i in json.load(open(FINAL / "annotations" / f"{s}.json", encoding="utf-8"))["images"] if not isov(i["file_name"])}
        fp = {i["file_name"] for i in json.load(open(PURE / "annotations" / f"{s}.json", encoding="utf-8"))["images"] if not isov(i["file_name"])}
        ok = ff == fp; allok &= ok
        print(f"   {'✅' if ok else '❌'} [{s}] {len(ff)} == {len(fp)}")

    fn = {s: {i["file_name"] for i in json.load(open(FINAL / "annotations" / f"{s}.json", encoding="utf-8"))["images"]} for s in ["train", "valid", "test"]}
    ov = (fn["train"] & fn["valid"]) | (fn["train"] & fn["test"]) | (fn["valid"] & fn["test"])
    allok &= not ov
    print(f"\n[跨 split 檔名重疊] {len(ov)} {'✅' if not ov else '❌'}")

    print("\n" + "=" * 60)
    print("🎉 全部通過，資料集可正式採用" if allok else "❌ 有項目未通過")
    print("=" * 60)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
