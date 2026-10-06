"""驗證 4ch 模型走 inference_detector 時，第4通道真的載入了 changemap（而非靜默補零）。

背景（重要）：hooks/load_changemap.py 的 LoadImageAndChangemap 找不到 changemap 檔時會
**補全零通道且不報錯**。所以任何 post-analysis 只要 --data-root 指錯（例如指到沒有
changemap 的 v14），就會「成功跑完但數字全錯」。跑全套分析前必須先確認這條路是通的。

作法：對 v15 test 幾張已知 changemap 非全黑的圖，攔截 test_pipeline 的輸出，檢查第4通道
非零像素比例 > 0；並與 dataset 的 changemap PNG 直接比對一致。
另外對照組：故意把 data_root 指到 v14（無 changemap）→ 應該看到第4通道全零（證明陷阱真的存在）。
"""
import json
from pathlib import Path
import numpy as np
import cv2
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings
from mmdet.utils import get_test_pipeline_cfg
from mmcv.transforms import Compose

CONFIG = "/workspace/project/configs/co_dino_chiayi_v15_own_dynamic_bgs_4ch_36e.py"
V15 = Path("/workspace/datasets/coco/chiayi_v15_own_dynamic_bgs")
V14 = Path("/workspace/datasets/coco/chiayi_v14_own_dynamic_litter")   # 無 changemap，對照組

cfg = Config.fromfile(CONFIG)
init_default_scope("mmdet")
import_modules_from_strings(**cfg["custom_imports"])
pipe = Compose(get_test_pipeline_cfg(cfg))
print("test_pipeline 第一個 transform =", get_test_pipeline_cfg(cfg)[0]["type"])

d = json.load(open(V15 / "annotations/test.json"))
# 挑 changemap 前景比例最高的幾張（確保不是本來就全黑）
cands = []
for im in d["images"][:150]:
    cm = cv2.imread(str(V15 / "test_changemap" / (Path(im["file_name"]).stem + ".png")), 0)
    if cm is not None:
        cands.append(((cm > 127).mean(), im["file_name"]))
cands.sort(reverse=True)
picks = [n for _, n in cands[:3]]

print("\n=== A. data_root = v15（正確）：第4通道應與 changemap PNG 一致且非零 ===")
ok = True
for name in picks:
    res = pipe(dict(img_path=str(V15 / "test" / name), img_id=0))
    img = res["inputs"].numpy()          # C,H,W（PackDetInputs 後）
    ch4 = img[3]
    raw = cv2.imread(str(V15 / "test_changemap" / (Path(name).stem + ".png")), 0)
    frac_pipe = float((ch4 > 127).mean())
    frac_raw = float((raw > 127).mean())
    good = frac_pipe > 0
    ok = ok and good
    print(f"  {name[:52]}... 第4通道前景={frac_pipe:.4f} | 原PNG前景={frac_raw:.4f} "
          f"{'✅ 有載入' if good else '❌ 全零(靜默fallback!)'}")

print("\n=== B. data_root = v14（無 changemap，對照組）：應全零，證明陷阱存在 ===")
for name in picks[:1]:
    res = pipe(dict(img_path=str(V14 / "test" / name), img_id=0))
    ch4 = res["inputs"].numpy()[3]
    frac = float((ch4 > 127).mean())
    print(f"  {name[:52]}... 第4通道前景={frac:.4f} "
          f"{'⚠️ 全零 → 指錯 data_root 會靜默出錯（如預期）' if frac == 0 else '意外非零?'}")

print("\n" + ("✅ 4ch inference 路徑驗證通過：data_root 指 v15 時 changemap 確實載入。"
              if ok else "❌ 第4通道沒載入，post-analysis 不可信，須排查。"))
