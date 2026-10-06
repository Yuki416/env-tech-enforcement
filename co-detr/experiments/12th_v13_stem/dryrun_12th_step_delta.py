"""釘死最後一個線頭：用真正的 optim_wrapper 跑 N 步，逐步量 conv1 第4通道權重位移。
若每步 ||Δ||>0 → 權重確實在更新，第一次 dry-run 的 Δ=0 是量法問題，可放心 launch。
"""
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope, MODELS
from mmengine.runner import Runner
from mmengine.optim import build_optim_wrapper
from mmengine.utils import import_modules_from_strings

CONFIG = "/workspace/project/configs/co_dino_chiayi_v13_4ch_litter_12th_36e.py"
dev = "cuda:0"

cfg = Config.fromfile(CONFIG)
init_default_scope("mmdet")
import_modules_from_strings(**cfg["custom_imports"])

model = MODELS.build(cfg.model); model.init_weights(); model.to(dev); model.train()
dl = dict(cfg.train_dataloader); dl["batch_size"]=1; dl["num_workers"]=0; dl["persistent_workers"]=False
loader = Runner.build_dataloader(dl)
optim_wrapper = build_optim_wrapper(model, cfg.optim_wrapper)

it = iter(loader)
prev = model.backbone.conv1.weight[:, 3].detach().clone()
for i in range(3):
    data = next(it)
    losses = model.train_step(data, optim_wrapper)   # 真正的訓練步驟（含 clip_grad + AdamW step）
    cur = model.backbone.conv1.weight[:, 3].detach()
    step_delta = (cur - prev).norm().item()
    print(f"[step {i}] conv1 ch4 本步位移 ||Δ||={step_delta:.6e}")
    prev = cur.clone()
print("DONE")
