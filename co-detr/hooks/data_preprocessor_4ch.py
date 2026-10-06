"""支援 4 通道 mean/std 的 DetDataPreprocessor。
MMEngine 內建 data_preprocessor 的 mean/std 只接受 1 或 3 個值，無法直接 4 通道。
作法：父類不傳 mean/std（停用內建 normalize），本類在父類做完 bgr/pad 後，自己用 4 通道
mean/std 做 normalize。"""
import torch
from mmdet.registry import MODELS
from mmdet.models.data_preprocessors import DetDataPreprocessor


@MODELS.register_module()
class Det4ChDataPreprocessor(DetDataPreprocessor):
    def __init__(self, mean, std, **kwargs):
        super().__init__(**kwargs)   # 不傳 mean/std → 父類 enable_normalize=False
        self.register_buffer('mean4', torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1), False)
        self.register_buffer('std4', torch.tensor(std, dtype=torch.float32).view(-1, 1, 1), False)

    def forward(self, data, training=False):
        data = super().forward(data, training)        # cast / bgr_to_rgb(False) / pad
        data['inputs'] = (data['inputs'].float() - self.mean4) / self.std4
        return data
