"""釐清 conv1 第4通道權重為何沒更新：是設定沒生效，還是 changemap 太稀疏導致該 batch 梯度=0。
做法：手動 preprocess → 看第4通道輸入有多少非零像素 → 手動 backward → 看 conv1.weight.grad
第4通道 vs RGB 的 L2 norm。跑 15 個 iteration。
"""
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope, MODELS
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings
from mmengine.structures import BaseDataElement

CONFIG = "/workspace/project/configs/co_dino_chiayi_v13_4ch_litter_12th_36e.py"
N = 15

cfg = Config.fromfile(CONFIG)
init_default_scope("mmdet")
import_modules_from_strings(**cfg["custom_imports"])

dev = "cuda:0"
model = MODELS.build(cfg.model); model.init_weights(); model.to(dev); model.train()

dl = dict(cfg.train_dataloader); dl["batch_size"]=1; dl["num_workers"]=0; dl["persistent_workers"]=False
loader = Runner.build_dataloader(dl)

print(f"{'iter':>4} {'ch4_nonzero_px':>14} {'grad_ch4_L2':>14} {'grad_rgb_L2':>14}")
it = iter(loader)
n_with_motion = 0
for i in range(N):
    data = next(it)
    data = model.data_preprocessor(data, training=True)
    inputs, samples = data["inputs"], data["data_samples"]
    nz = int((inputs[:, 3].abs() > 1e-6).sum().item())     # 第4通道(已正規化)非零像素數
    n_with_motion += (nz > 0)
    model.zero_grad()
    losses = model.loss(inputs, samples)
    total, _ = model.parse_losses(losses)
    total.backward()
    g = model.backbone.conv1.weight.grad
    g4 = g[:, 3].norm().item()
    grgb = g[:, :3].norm().item()
    print(f"{i:>4} {nz:>14} {g4:>14.6e} {grgb:>14.6e}")

print(f"\n{N} 個 batch 中有 motion 像素的：{n_with_motion} 個")
print("判讀：若『有 motion 像素的 batch』其 grad_ch4_L2 > 0 → 設定正確，反傳確實流到第4通道；")
print("      grad_ch4 只在 ch4_nonzero=0 時為 0 → 純粹是 changemap 稀疏，不是 bug。")
print("DONE")
