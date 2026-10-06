#!/usr/bin/env python3
"""用 Python zipfile 全量解壓 v17，並逐 split 比對磁碟檔名 vs JSON。

背景（第 15 次踩過的坑，見 15th README「解壓必須用 Python zipfile」）：
系統 `unzip` CLI 會拿 local file header 猜編碼，把含中文的檔名洗成西里爾亂碼
（例：中文地名 → <亂碼>），且只噴警告不報錯，
若沒逐一比對 JSON 就會靜默拿到壞資料集。Python zipfile 依 UTF-8 flag 正確解碼。

本次一開始就用 zipfile 全量解壓（不碰系統 unzip），解完對三個 split 都做
「JSON file_name 集合 == 磁碟檔名集合」的驗證，任一 split 不吻合就 exit 1。
"""
import json
import shutil
import sys
import zipfile
from pathlib import Path

COCO = Path("/path/to/data/coco")
NAME = "chiayi_v17_garbage_litter_refined_122vids"
ZIP = COCO / f"{NAME}.zip"
DST = COCO / NAME
SPLITS = ["train", "valid", "test"]

if not ZIP.exists():
    sys.exit(f"❌ 找不到 zip：{ZIP}")

if DST.exists():
    print(f"刪除舊的解壓結果 {DST} ...", flush=True)
    shutil.rmtree(DST)
DST.mkdir(parents=True)

z = zipfile.ZipFile(ZIP)
members = z.namelist()
print(f"zip 內共 {len(members)} 個項目，開始全量解壓 ...", flush=True)

# 逐一解壓（用 zipfile 的 UTF-8 解碼，不用系統 unzip）
top = f"{NAME}/"
n_done = 0
for n in members:
    if n.endswith("/"):
        continue
    rel = n[len(top):] if n.startswith(top) else n
    out_path = DST / rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with z.open(n) as src, open(out_path, "wb") as out:
        shutil.copyfileobj(src, out)
    n_done += 1
    if n_done % 1000 == 0:
        print(f"  {n_done} ...", flush=True)
print(f"解壓完成，共 {n_done} 個檔案。", flush=True)

# ---- 驗證：逐 split 比對 JSON file_name vs 磁碟 ----
print("\n=== 驗證（JSON file_name vs 磁碟檔名）===", flush=True)
all_ok = True
for s in SPLITS:
    ann = DST / "annotations" / f"{s}.json"
    if not ann.exists():
        print(f"  [{s}] ❌ 找不到 {ann}")
        all_ok = False
        continue
    want = {i["file_name"] for i in json.load(open(ann, encoding="utf-8"))["images"]}
    img_dir = DST / s
    have = {p.name for p in img_dir.iterdir()} if img_dir.exists() else set()
    miss, extra = want - have, have - want
    status = "✅" if not (miss or extra) else "❌"
    print(f"  [{s}] {status}  JSON 要 {len(want)}  磁碟有 {len(have)}  缺少 {len(miss)}  多餘 {len(extra)}")
    if miss:
        print(f"        缺例: {list(miss)[:2]}")
    if extra:
        print(f"        多例: {list(extra)[:2]}")
    if miss or extra:
        all_ok = False

if not all_ok:
    print("\n❌ 有 split 不吻合，請勿使用此解壓結果。")
    sys.exit(1)
print("\n✅ 三個 split 檔名與 JSON 完全吻合。")
