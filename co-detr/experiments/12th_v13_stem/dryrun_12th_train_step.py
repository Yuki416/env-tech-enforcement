"""第 12 次訓練 launch 前 dry-run：跑真實的 forward+backward(train_step)，
驗證 (a) 解凍 stem 後不 OOM、(b) 4ch 輸入整條訓練步驟可跑、(c) 反傳真的流過 conv1。
bs=1、num_workers=0，跑 3 個 iteration，回報 peak GPU memory。
"""
import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope, MODELS
from mmengine.runner import Runner
from mmengine.optim import build_optim_wrapper
from mmengine.utils import import_modules_from_strings

CONFIG = "/workspace/project/configs/co_dino_chiayi_v13_4ch_litter_12th_36e.py"
N_ITERS = 3

cfg = Config.fromfile(CONFIG)
init_default_scope("mmdet")
import_modules_from_strings(**cfg["custom_imports"])

device = "cuda:0"
model = MODELS.build(cfg.model)
model.init_weights()
model.to(device)
model.train()

# bs=1、單線程 loader（與實際訓練同一條 4ch pipeline）
dl_cfg = dict(cfg.train_dataloader)
dl_cfg["batch_size"] = 1
dl_cfg["num_workers"] = 0
dl_cfg["persistent_workers"] = False
loader = Runner.build_dataloader(dl_cfg)

optim_wrapper = build_optim_wrapper(model, cfg.optim_wrapper)

# 記錄 conv1 第4通道初值，跑完看有沒有被更新（反傳是否真的流過 stem）
w0 = model.backbone.conv1.weight[:, 3].detach().clone()

torch.cuda.reset_peak_memory_stats(device)
it = iter(loader)
for i in range(N_ITERS):
    data = next(it)
    losses = model.train_step(data, optim_wrapper)  # preprocess+forward+backward+step
    total = sum(v.item() for v in losses.values() if v.ndim == 0)
    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    print(f"[iter {i}] total_loss≈{total:.3f}  peak_mem={peak:.2f} GB")

w1 = model.backbone.conv1.weight[:, 3].detach()
delta = (w1 - w0).norm().item()
print(f"\nconv1 第4通道權重變化量 ||Δ||={delta:.6e}  (>0 代表反傳有更新到 stem/遮罩通道)")
print(f"最終 peak GPU memory = {torch.cuda.max_memory_allocated(device)/1024**3:.2f} GB")
print("DRYRUN_OK")
