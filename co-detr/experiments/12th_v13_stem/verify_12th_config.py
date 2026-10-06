"""驗證第 12 次 config：frozen_stages/conv1 通道數/paramwise lr 分組是否正確。"""
from mmengine.config import Config
from mmengine.registry import init_default_scope, MODELS, OPTIM_WRAPPERS
from mmengine.utils import import_modules_from_strings

CONFIG = "/workspace/project/configs/co_dino_chiayi_v13_4ch_litter_12th_36e.py"

cfg = Config.fromfile(CONFIG)
init_default_scope("mmdet")
import_modules_from_strings(**cfg["custom_imports"])

print("=== config 解析 OK ===")
print("backbone.frozen_stages =", cfg.model.backbone.frozen_stages)
print("backbone.norm_eval     =", cfg.model.backbone.norm_eval)
print("backbone.in_channels   =", cfg.model.backbone.in_channels)
print("paramwise custom_keys  =", cfg.optim_wrapper.paramwise_cfg["custom_keys"])

model = MODELS.build(cfg.model)
model.init_weights()
w = model.backbone.conv1.weight
print("\nconv1.weight shape     =", tuple(w.shape), "(應為 [64,4,7,7])")
print("conv1.weight.requires_grad =", w.requires_grad, "(應為 True)")

# 建 optim_wrapper（用 mmengine 標準 helper，會套用 paramwise_cfg）
from mmengine.optim import build_optim_wrapper
optim_wrapper = build_optim_wrapper(model, cfg.optim_wrapper)
opt = optim_wrapper.optimizer

# 對照 param id → name
id2name = {id(p): n for n, p in model.named_parameters()}
base_lr = cfg.optim_wrapper.optimizer["lr"]
conv1_lrs, backbone_other_lrs, head_lrs = [], [], []
for g in opt.param_groups:
    for p in g["params"]:
        n = id2name.get(id(p), "?")
        if n.startswith("backbone.conv1"):
            conv1_lrs.append((n, g["lr"]))
        elif n.startswith("backbone."):
            backbone_other_lrs.append(g["lr"])

print("\n=== lr 分組驗證 (base lr =", base_lr, ") ===")
for n, lr in conv1_lrs:
    print(f"  [conv1] {n:35s} lr={lr:.2e}  (期望 {base_lr:.0e} = 1.0x)")
if backbone_other_lrs:
    s = set(f"{x:.2e}" for x in backbone_other_lrs)
    print(f"  [backbone 其餘] {len(backbone_other_lrs)} 個 param，lr 集合={s}  (期望 {base_lr*0.1:.0e} = 0.1x)")
print("\n結論：若 conv1 lr=1e-04、其餘 backbone lr=1e-05、conv1 shape=[64,4,7,7] 且 requires_grad=True → 設定正確。")
