"""變因① 靜止尾處理組 dry-run：確認 config 可載入、資料讀得到、5 類、negative 保留、
ClassBalancedDataset 未過採樣、max_keep_ckpts=-1（全留）、一個 train batch 能 forward(loss) 無 NaN。不訓練。

用法（docker 內）：
  PYTHONUTF8=1 python3 /workspace/project/experiments/ablation_study_4vars_20260804/dryrun_var1_statictail.py
"""
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmdet.registry import MODELS
import torch

init_default_scope('mmdet')
import projects  # noqa  Co-DETR registry

CFG = '/workspace/project/configs/co_dino_chiayi_v17_statictail_ablation_36e.py'
cfg = Config.fromfile(CFG)
cfg.train_dataloader.batch_size = 1
cfg.val_dataloader.batch_size = 1

print('data_root      :', cfg.data_root)
print('classes        :', cfg.CLASSES)
print('filter_empty_gt:', cfg.train_dataloader.dataset.dataset.filter_cfg)
print('oversample_thr :', cfg.train_dataloader.dataset.oversample_thr)
print('work_dir       :', cfg.work_dir)
print('save_best      :', cfg.default_hooks.checkpoint.save_best)
print('max_keep_ckpts :', cfg.default_hooks.checkpoint.max_keep_ckpts)
print()

# ── max_keep_ckpts 必須 -1（全留 36 顆，畫完整 test-static 曲線）──────────────
assert cfg.default_hooks.checkpoint.max_keep_ckpts == -1, (
    f'❌ max_keep_ckpts 應為 -1（全留），實得 {cfg.default_hooks.checkpoint.max_keep_ckpts}')
print('✅ max_keep_ckpts=-1（全留 36 顆）')

# ── 資料載入 ──────────────────────────────────────────────
train_dl = Runner.build_dataloader(cfg.train_dataloader)
val_dl = Runner.build_dataloader(cfg.val_dataloader)
inner = train_dl.dataset.dataset
n_inner = len(inner)
n_wrapped = len(train_dl.dataset)
print('train CocoDataset 幀數      :', n_inner, '(含 negative)')
print('train ClassBalanced 展開後  :', n_wrapped)
print('valid 幀數                  :', len(val_dl.dataset))

try:
    gar_imgs = sum(
        1 for i in range(n_inner)
        if any(inst.get('bbox_label') == 4 for inst in inner.get_data_info(i).get('instances', []))
    )
    print(f'garbage 影像佔比 f          : {gar_imgs}/{n_inner} = {gar_imgs/n_inner:.4f}')
except Exception as e:
    print(f'garbage 影像佔比            : （略過：{e}）')
print()

# ── ★ 關鍵 assert：沒有過採樣 ─────────────────────────────
assert n_wrapped == n_inner, (
    f'❌ ClassBalancedDataset 觸發過採樣！{n_inner} → {n_wrapped}。oversample_thr 應為 0.0')
print(f'✅ 無過採樣：ClassBalanced 展開後 == CocoDataset（皆 {n_inner}）')

# ── 規模檢查：處理組定案值 ────────────────────────────────
assert n_inner == 3551, f'❌ train 應為 3551 幀（處理組），實得 {n_inner}'
assert len(val_dl.dataset) == 373, f'❌ valid 應為 373 幀（= 16th），實得 {len(val_dl.dataset)}'
print('✅ 規模正確：train 3551 / valid 373')
print()

batch = next(iter(train_dl))

# ── 建模型 + forward(loss) ────────────────────────────────
model = MODELS.build(cfg.model).cuda()
model.train()
data = model.data_preprocessor(batch, True)
print('輸入張量 shape:', tuple(data['inputs'].shape), '(應為 [1,3,H,W])')
assert data['inputs'].shape[1] == 3, '❌ 通道數不是 3（純 RGB）!'
losses = model(**data, mode='loss')
finite = all(torch.isfinite(v).all() for v in losses.values() if torch.is_tensor(v))
print('loss 數量:', len(losses), '| keys:', list(losses.keys())[:4], '...')
print('所有 loss 有限:', finite)
print('DRY-RUN OK ✓' if finite else 'DRY-RUN 有 NaN/Inf ✗')
