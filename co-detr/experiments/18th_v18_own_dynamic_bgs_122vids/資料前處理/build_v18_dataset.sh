#!/usr/bin/env bash
# 第18次 v18 資料集 build（步驟2）：整份 copy v17 手動版 → v18，放入評估用標註。
# v18 = v17 手動版(3 split 影像+標註) 的實體複製 + (稍後步驟3) 平行 changemap。
# 使用者定案：影像用 copy（非 symlink）、三 split 全要、不動 v17。
#
# 用法（背景）：
#   nohup bash experiments/18th_v18_own_dynamic_bgs_122vids/資料前處理/build_v18_dataset.sh \
#     > /path/to/data/coco/build_v18_dataset.log 2>&1 &
#   tail -f /path/to/data/coco/build_v18_dataset.log
set -euo pipefail

SRC=/path/to/data/coco/chiayi_v17_own_dynamic_litter_head10_manual
V14=/path/to/data/coco/chiayi_v14_own_dynamic_litter
DST=/path/to/data/coco/chiayi_v18_own_dynamic_bgs_122vids

echo "=== [1/4] 建目標資料夾 ==="
mkdir -p "$DST/annotations"

echo "=== [2/4] rsync 複製 3 split 影像（copy，非 symlink）==="
for s in train valid test; do
    echo "  -- $s --"
    rsync -a "$SRC/$s/" "$DST/$s/"
done

echo "=== [3/4] 複製標註 ==="
# v17 手動版原有 4 個標註
for f in train.json valid.json test.json test_static594.json; do
    cp -f "$SRC/annotations/$f" "$DST/annotations/$f"
    echo "  copied $f"
done
# 動態子集（沿用 v14 既有、與 v17 test 逐框 GT 一致）
cp -f "$V14/annotations/test_dynamic166.json" "$DST/annotations/test_dynamic166.json"
echo "  copied test_dynamic166.json (from v14)"

echo "=== [4/4] 驗證 ==="
python3 - "$SRC" "$DST" <<'PY'
import json, os, sys
SRC, DST = sys.argv[1], sys.argv[2]
ok = True

# 影像數對得上
for s in ("train", "valid", "test"):
    a = len(os.listdir(os.path.join(SRC, s)))
    b = len(os.listdir(os.path.join(DST, s)))
    tag = "OK" if a == b else "❌"
    if a != b: ok = False
    print(f"  [{tag}] {s} 影像數 src={a} dst={b}")

# 每個標註的圖片都落盤在對應 split
def check_ann(ann, split):
    global ok
    j = json.load(open(os.path.join(DST, "annotations", ann)))
    fns = [i["file_name"] for i in j["images"]]
    miss = [f for f in fns if not os.path.exists(os.path.join(DST, split, f))]
    tag = "OK" if not miss else "❌"
    if miss: ok = False
    print(f"  [{tag}] {ann}: {len(fns)} 張標註圖，缺檔 {len(miss)}")

for ann, split in [("train.json","train"),("valid.json","valid"),
                   ("test.json","test"),("test_static594.json","test"),
                   ("test_dynamic166.json","test")]:
    check_ann(ann, split)

# dynamic166 ⊆ test，且與 v17 test GT 逐框一致（再驗一次）
d166 = json.load(open(os.path.join(DST,"annotations","test_dynamic166.json")))
tj   = json.load(open(os.path.join(DST,"annotations","test.json")))
def by_fn(j):
    id2fn={i["id"]:i["file_name"] for i in j["images"]}
    d={}
    for a in j["annotations"]:
        d.setdefault(id2fn[a["image_id"]],[]).append((a["category_id"],tuple(round(x,1) for x in a["bbox"])))
    return {k:sorted(v) for k,v in d.items()}
a,b = by_fn(d166), by_fn(tj)
mism = sum(1 for fn in a if b.get(fn)!=a[fn])
tag = "OK" if mism==0 else "❌"
if mism: ok=False
print(f"  [{tag}] dynamic166 vs test.json GT 逐框一致：不一致 {mism}/{len(a)}")

print("\n=== BUILD_V18_" + ("OK" if ok else "FAILED") + " ===")
sys.exit(0 if ok else 1)
PY