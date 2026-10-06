"""4 通道載入 transform：RGB + changemap → HxWx4。
RGB 以 RGB 順序載入（data_preprocessor 設 bgr_to_rgb=False）；
changemap 由 RGB 路徑推導：<split>/X.jpg -> <split>_changemap/X.png。
找不到 changemap 時補零通道（等同無 motion）。"""
from pathlib import Path

import numpy as np
import mmcv
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadImageAndChangemap(BaseTransform):
    def __init__(self, to_float32=False, changemap_suffix='_changemap'):
        self.to_float32 = to_float32
        self.changemap_suffix = changemap_suffix

    def transform(self, results):
        img_path = results['img_path']
        img = mmcv.imread(img_path, channel_order='rgb')          # HxWx3 (RGB)
        p = Path(img_path)
        cm_path = p.parent.parent / (p.parent.name + self.changemap_suffix) / (p.stem + '.png')
        cm = mmcv.imread(str(cm_path), flag='grayscale') if cm_path.exists() else None
        if cm is None:
            cm = np.zeros(img.shape[:2], dtype=img.dtype)         # 無 changemap -> 零通道
        elif cm.shape[:2] != img.shape[:2]:
            cm = mmcv.imresize(cm, (img.shape[1], img.shape[0]))
        img4 = np.concatenate([img, cm[..., None]], axis=2)       # HxWx4
        if self.to_float32:
            img4 = img4.astype(np.float32)
        results['img'] = img4
        results['img_shape'] = img4.shape[:2]
        results['ori_shape'] = img4.shape[:2]
        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(changemap_suffix={self.changemap_suffix})'
