#!/usr/bin/env python3
"""用 Python zipfile 重新解壓 v16 的 train split（修復系統 unzip 造成的檔名亂碼）。

背景：系統 `unzip` CLI 會拿 local file header 去猜編碼，把 1108 個含中文的 train 檔名
洗成西里爾字母亂碼（例：中文地名 → <亂碼>）。
v16 README 已警告「不可用檔案總管或 7-Zip 解壓」—— Linux 的 unzip CLI 同樣中招。
Python zipfile 會依 UTF-8 flag 正確解碼，實測 7347/7347 檔名與 JSON 完全吻合。

valid/test 經檢查無亂碼（0 mismatch），故只重解 train。
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

COCO = Path("/path/to/data/coco")
ZIP = COCO / "chiayi_v16_garbage_litter_121vids.zip"
DST = COCO / "chiayi_v16_garbage_litter_121vids"
SPLIT = "train"

want = {i["file_name"] for i in
        json.load(open(DST / "annotations" / f"{SPLIT}.json", encoding="utf-8"))["images"]}
print(f"JSON 要求 {len(want)} 張", flush=True)

tgt = DST / SPLIT
if tgt.exists():
    print(f"刪除舊的（含亂碼）{tgt} ...", flush=True)
    shutil.rmtree(tgt)
tgt.mkdir(parents=True)

z = zipfile.ZipFile(ZIP)
members = [n for n in z.namelist() if n.startswith(f"{DST.name}/{SPLIT}/") and n.endswith(".jpg")]
print(f"zip 內 {SPLIT} jpg: {len(members)} 張，開始解壓 ...", flush=True)

for i, n in enumerate(members, 1):
    with z.open(n) as src, open(tgt / Path(n).name, "wb") as out:
        shutil.copyfileobj(src, out)
    if i % 500 == 0:
        print(f"  {i}/{len(members)}", flush=True)

have = {p.name for p in tgt.iterdir()}
miss, extra = want - have, have - want
print(f"\n=== 驗證 ===")
print(f"  JSON 要 {len(want)}  磁碟有 {len(have)}")
print(f"  缺少: {len(miss)}   多餘: {len(extra)}")
if miss or extra:
    print(f"  ❌ 仍不吻合  缺例={list(miss)[:2]}  多例={list(extra)[:2]}")
    sys.exit(1)
print("  ✅ 檔名與 JSON 完全吻合")
