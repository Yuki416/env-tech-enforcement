"""目視檢查：增強後 RGB 與 changemap 是否對齊（早期融合最隱形的 bug）。

跑「訓練 pipeline」於前幾張 train 影像，把增強後的第4通道(changemap)以紅色半透明
疊回增強後的 RGB，存成 PNG。若紅色區域沒對到畫面中該動的位置 → 同步出問題。

用法（容器內）：
  python /workspace/project/viz_check_4ch_alignment.py --num 8 \
      --out /workspace/hdd2_storage/12th_expt_v13_bgs_4ch_stem_unfreeze_36e/align_check
"""
import argparse
from pathlib import Path

import numpy as np
import cv2
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmcv.transforms import Compose
from mmengine.utils import import_modules_from_strings

CONFIG = "/workspace/project/configs/co_dino_chiayi_v13_4ch_litter_12th_36e.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG)
    ap.add_argument("--num", type=int, default=8)
    ap.add_argument("--out", default="/workspace/hdd2_storage/"
                    "12th_expt_v13_bgs_4ch_stem_unfreeze_36e/align_check")
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope("mmdet")
    if cfg.get("custom_imports"):
        import_modules_from_strings(**cfg["custom_imports"])

    # 用訓練 pipeline，但去掉 PackDetInputs（直接拿 results['img']）
    pipeline = Compose([t for t in cfg.load_pipeline if t["type"] != "PackDetInputs"])

    import json
    data_root = Path(cfg.data_root)
    ann = json.loads((data_root / "annotations/train.json").read_text())
    imgs = ann["images"][: args.num]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for im in imgs:
        img_path = str(data_root / "train" / im["file_name"])
        res = pipeline(dict(img_path=img_path, img_id=im["id"],
                            instances=[]))
        img4 = res["img"]                          # HxWx4, RGB+changemap
        rgb = img4[..., :3].astype(np.uint8)
        cm = img4[..., 3].astype(np.uint8)
        # changemap 以紅色疊上
        overlay = rgb.copy()
        mask = cm > 127
        overlay[mask] = (0.5 * overlay[mask] +
                         0.5 * np.array([255, 0, 0])).astype(np.uint8)
        # 並排：左=增強後 RGB，右=疊圖
        canvas = np.concatenate([rgb, overlay], axis=1)
        # 存成 BGR
        out_path = out_dir / (Path(im["file_name"]).stem + "_align.png")
        cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        print(f"[saved] {out_path}  (cm>127 px={int(mask.sum())})")

    print(f"\n完成。{len(imgs)} 張存於 {out_dir}")
    print("檢查：右圖紅色（motion）應落在畫面中正在動的位置，且與左圖幾何一致。")


if __name__ == "__main__":
    main()
