"""第 18 次 v18 4ch config 驗證 + 單步 dryrun（不啟動正式訓練）。
  A. config 解析、conv1 shape/requires_grad、paramwise lr 分組
     （conv1 1e-4 / backbone 其餘 1e-6[=0.01x,同16th] / head 1e-4）
  B. oversample_thr=0.0 → len(ClassBalancedDataset)==len(CocoDataset)（防依分布觸發過採樣）
  C. 真的過 train batch：輸入 4 通道、第4通道非全零(changemap 真的讀到)、前向+反向不 OOM/NaN
  D. val predict 路徑
容器內執行（GPU）。"""
import math, torch
from mmengine.config import Config
from mmengine.registry import init_default_scope, MODELS
from mmengine.utils import import_modules_from_strings
from mmengine.optim import build_optim_wrapper
from mmengine.runner import Runner

CONFIG = "/workspace/project/configs/co_dino_chiayi_v18_own_dynamic_bgs_4ch_36e.py"

cfg = Config.fromfile(CONFIG)
init_default_scope("mmdet")
import_modules_from_strings(**cfg["custom_imports"])
base = cfg.optim_wrapper.optimizer["lr"]

print("=== A. config / lr 分組 ===")
print("data_root              =", cfg.data_root)
print("backbone.in_channels   =", cfg.model.backbone.in_channels, "(應 4)")
print("backbone.frozen_stages =", cfg.model.backbone.frozen_stages, "(應 -1)")
print("backbone.norm_eval     =", cfg.model.backbone.norm_eval, "(應 True)")
print("data_preprocessor      =", cfg.model.data_preprocessor["type"])
print("load_from              =", cfg.load_from.split("/")[-1])

model = MODELS.build(cfg.model)
model.init_weights()
w = model.backbone.conv1.weight
print("conv1.weight shape     =", tuple(w.shape), "(應 [64,4,7,7])")
print("conv1.requires_grad    =", w.requires_grad, "(應 True)")

optim_wrapper = build_optim_wrapper(model, cfg.optim_wrapper)
id2name = {id(p): n for n, p in model.named_parameters()}
conv1_lr, bb_other, head_lr = [], set(), set()
for g in optim_wrapper.optimizer.param_groups:
    for p in g["params"]:
        n = id2name.get(id(p), "?")
        if n.startswith("backbone.conv1"): conv1_lr.append((n, g["lr"]))
        elif n.startswith("backbone."): bb_other.add(f"{g['lr']:.2e}")
        elif any(n.startswith(h) for h in ("query_head", "roi_head", "bbox_head")): head_lr.add(f"{g['lr']:.2e}")
for n, lr in conv1_lr: print(f"  conv1: {n:38s} lr={lr:.2e} (期望 {base:.0e})")
print(f"  backbone 其餘 lr 集合={bb_other} (期望 {base*0.01:.0e} = 0.01x)")
print(f"  head lr 集合={head_lr} (期望 {base:.0e})")
lr_ok = (all(abs(lr-base)<1e-12 for _,lr in conv1_lr)
         and bb_other == {f"{base*0.01:.2e}"} and head_lr == {f"{base:.2e}"})
print("  lr 分組正確" if lr_ok else "  ❌ lr 分組不符期望")

print("\n=== B. oversample_thr=0.0 → 不過採樣 assert ===")
cfg.train_dataloader["batch_size"] = 1
cfg.train_dataloader["num_workers"] = 0
cfg.train_dataloader["persistent_workers"] = False
loader = Runner.build_dataloader(cfg.train_dataloader)
n_cbd = len(loader.dataset)
n_inner = len(loader.dataset.dataset)
os_ok = (n_cbd == n_inner)
print(f"  len(ClassBalancedDataset)={n_cbd}  len(CocoDataset)={n_inner}  "
      + ("相等→無過採樣 ✅" if os_ok else "❌ 不等→發生過採樣！"))

print("\n=== C. 真實 train_step dryrun（4ch 輸入 + changemap 非全零 + loss/梯度有限）===")
model = model.cuda().train()
optim_wrapper.initialize_count_status(model, 0, 3)
it = iter(loader)
ok = True; cm_ok = False
for i in range(3):
    batch = next(it)
    with optim_wrapper.optim_context(model):
        data = model.data_preprocessor(batch, training=True)
        if i == 0:
            C = data["inputs"].shape[1]
            print("  輸入張量 shape =", tuple(data["inputs"].shape), "(C 應 4)")
            assert C == 4, "輸入不是 4 通道！"
            # 第4通道非全零 → changemap 真的讀到（路徑對）
            frac = (data["inputs"][:, 3].abs() > 1e-6).float().mean().item()
            cm_ok = frac > 0.0
            print(f"  第4通道(changemap)非零像素比 = {frac*100:.2f}%  "
                  + ("→ changemap 有讀到 ✅" if cm_ok else "❌ 全零！changemap 路徑可能錯"))
        losses = model._run_forward(data, mode="loss")
    parsed, log_vars = model.parse_losses(losses)
    optim_wrapper.backward(parsed)
    g4 = model.backbone.conv1.weight.grad[:, 3].norm().item()
    optim_wrapper.step(); optim_wrapper.zero_grad()
    loss = float(log_vars["loss"])
    fin = math.isfinite(loss) and math.isfinite(g4)
    ok = ok and fin
    print(f"  iter{i}: loss={loss:.3f}  conv1_ch4_grad_L2={g4:.4f}  finite={fin}")
print(f"  GPU peak mem(train,bs=1) = {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

print("\n=== D. 驗證路徑 dryrun（predict 模式）===")
val_ok = True
try:
    val_loader = Runner.build_dataloader(cfg.val_dataloader)
    model.eval()
    with torch.no_grad():
        vb = next(iter(val_loader))
        vdata = model.data_preprocessor(vb, training=False)
        print("  val 輸入 shape =", tuple(vdata["inputs"].shape), "(C 應 4)")
        assert vdata["inputs"].shape[1] == 4
        preds = model._run_forward(vdata, mode="predict")
    print(f"  val forward OK，第一張預測 {len(preds[0].pred_instances.bboxes)} 個 box")
except Exception as e:
    val_ok = False
    import traceback; print("  ❌ 驗證路徑失敗:", type(e).__name__); traceback.print_exc()

all_ok = lr_ok and os_ok and ok and cm_ok and val_ok
print("\n" + ("✅ DRYRUN_ALL_PASS：4ch/lr/oversample/changemap/val 全通過 → 可正式訓練。"
              if all_ok else "❌ DRYRUN_FAIL：有問題，勿啟動訓練。"))
