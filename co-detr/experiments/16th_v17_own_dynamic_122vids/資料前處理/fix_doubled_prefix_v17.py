#!/usr/bin/env python3
"""修掉 v17 來源資料集中「clip 前綴重複」的檔名（磁碟 + JSON 一起改）。

背景：v17 新增的 valid clip（Day_Garbage_16_8_loc16_00m19s，來自「午後」Roboflow
專案，內部檔名格式與其他 clip 不同）在建置 step1 被前置了兩次 clip 名，產生：
    {clip}_{clip}_frame_XXXX.jpg
的重複前綴。JSON file_name 與磁碟一致（final_check 有過），不影響訓練，但命名醜、易混淆。

⚠️ 本腳本會就地修改本機解壓的來源資料集
   /path/to/data/coco/chiayi_v17_garbage_litter_refined_122vids/
   → 改完這些檔名就與 Google Drive 上的 zip 不一致（但磁碟↔JSON 仍自洽）。

作法：通用偵測三個 split 中 clip == P + '_' + P（前綴恰好重複一次）的影像，
把 {P}_{P}_frame_XXXX.jpg 改回 {P}_frame_XXXX.jpg，同步改磁碟檔名與 JSON file_name，
改前檢查目標名不存在（防撞名），改後重新比對該 split 磁碟↔JSON。
"""
import json
import re
import sys
from pathlib import Path

SRC = Path("/path/to/data/coco/chiayi_v17_garbage_litter_refined_122vids")
SPLITS = ["train", "valid", "test"]
frame_re = re.compile(r"^(.*)_frame_(\d+)\.jpg$", re.IGNORECASE)


def doubled_prefix(clip):
    """若 clip == P + '_' + P 回傳 P，否則 None。"""
    if clip is None:
        return None
    n = len(clip)
    if n % 2 == 0:  # 需 len = 2*len(P)+1 → 奇數
        return None
    half = (n - 1) // 2
    if clip[half] == "_" and clip[:half] == clip[half + 1:]:
        return clip[:half]
    return None


def fix_split(split):
    ann_path = SRC / "annotations" / f"{split}.json"
    j = json.load(open(ann_path, encoding="utf-8"))
    img_dir = SRC / split

    renames = []  # (old_name, new_name)
    for im in j["images"]:
        m = frame_re.match(im["file_name"])
        if not m:
            continue
        clip, frame = m.group(1), m.group(2)
        P = doubled_prefix(clip)
        if P is None:
            continue
        new_name = f"{P}_frame_{frame}.jpg"
        renames.append((im["file_name"], new_name, im))

    if not renames:
        print(f"  [{split}] 無前綴重複，跳過")
        return 0

    print(f"  [{split}] 發現 {len(renames)} 張前綴重複，開始修復 ...")
    # 防撞名：新名不得已存在（於磁碟或 JSON 其他項）
    existing = {im["file_name"] for im in j["images"]}
    for old, new, _ in renames:
        if new in existing and new != old:
            sys.exit(f"❌ 目標名已存在，恐撞名：{new}")
        if (img_dir / new).exists():
            sys.exit(f"❌ 磁碟已有目標檔：{new}")

    # 改磁碟 + JSON
    for old, new, im in renames:
        (img_dir / old).rename(img_dir / new)
        im["file_name"] = new

    json.dump(j, open(ann_path, "w", encoding="utf-8"), ensure_ascii=False)

    # 重新驗證該 split 磁碟↔JSON
    want = {im["file_name"] for im in j["images"]}
    have = {p.name for p in img_dir.iterdir()}
    miss, extra = want - have, have - want
    ok = not (miss or extra)
    print(f"    改完驗證：JSON {len(want)} / 磁碟 {len(have)}  缺 {len(miss)} 多 {len(extra)}  "
          + ("✅" if ok else "❌"))
    if not ok:
        print(f"      缺例={list(miss)[:2]}  多例={list(extra)[:2]}")
        sys.exit(1)
    # 展示一筆修復前後
    o, n, _ = renames[0]
    print(f"    範例: {o}\n       → {n}")
    return len(renames)


if __name__ == "__main__":
    print(f"修復來源：{SRC}\n")
    total = sum(fix_split(s) for s in SPLITS)
    print(f"\n完成，共修復 {total} 張。")
