#!/usr/bin/env python3
"""建立「166 張動態子集」的 5 類標註（輔助判定用）。

背景：主判定是 594 靜止 test。輔助判定要問「BGS 在它**有訊號**的地方（垃圾仍在動）表現如何」。
那 166 張動態幀 = 第 11/12 次用的 v13 動態 test，已驗證是我們 594 test 的**真子集**。
但 v13 的標註是「其他組別、單類別 litter」→ 不能用。這裡改從 **v15/v14 自己的 5 類 test.json**
過濾出那 166 個檔名，得到「同一批動態影像 + 我們自己的 5 類 GT」。

輸出（內容相同，兩份，讓 v15 與 v14 兩個 checkpoint 都能用各自 data_root 評估）：
  chiayi_v15_own_dynamic_bgs/annotations/test_dynamic166.json
  chiayi_v14_own_dynamic_litter/annotations/test_dynamic166.json
"""
import json
from pathlib import Path

V13_CM = Path("/path/to/data/coco/chiayi_v13_other_team_litter_dynamic_test_frame/test_changemap")
V15 = Path("/path/to/data/coco/chiayi_v15_own_dynamic_bgs")
V14 = Path("/path/to/data/coco/chiayi_v14_own_dynamic_litter")
OUT_NAME = "test_dynamic166.json"

dyn_stems = {p.stem for p in V13_CM.glob("*.png")}
print(f"v13 動態 test 幀數 = {len(dyn_stems)}")

src = json.load(open(V15 / "annotations/test.json"))
keep_imgs = [im for im in src["images"] if Path(im["file_name"]).stem in dyn_stems]
keep_ids = {im["id"] for im in keep_imgs}
keep_anns = [a for a in src["annotations"] if a["image_id"] in keep_ids]

missing = dyn_stems - {Path(im["file_name"]).stem for im in keep_imgs}
if missing:
    raise SystemExit(f"❌ 有 {len(missing)} 個動態幀不在 594 test 中，子集假設不成立：{list(missing)[:3]}")

out = dict(
    images=keep_imgs,
    annotations=keep_anns,
    categories=src["categories"],
)
CATS = {c["id"]: c["name"] for c in src["categories"]}
cnt = {}
for a in keep_anns:
    cnt[CATS[a["category_id"]]] = cnt.get(CATS[a["category_id"]], 0) + 1

for root in (V15, V14):
    p = root / "annotations" / OUT_NAME
    json.dump(out, open(p, "w"), ensure_ascii=False)
    print(f"寫出 {p}")

print(f"\n✅ 動態子集：{len(keep_imgs)} 張影像 / {len(keep_anns)} 個框（應為 166 張）")
print("   各類框數:", cnt)
