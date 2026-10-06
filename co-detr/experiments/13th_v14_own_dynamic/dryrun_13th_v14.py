"""v14 動態版 dry-run：確認 config 可載入、v14 資料讀得到、5 類對齊、negative 保留、
一個 train batch 能 forward(loss) 且無 NaN。不進行實際訓練。"""
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmdet.registry import MODELS
import torch

init_default_scope('mmdet')
import projects  # noqa  Co-DETR registry (PYTHONPATH /opt/mmdetection)

CFG = '/workspace/project/configs/co_dino_chiayi_v14_own_dynamic_36e.py'
cfg = Config.fromfile(CFG)
cfg.train_dataloader.batch_size = 1
cfg.val_dataloader.batch_size = 1

print('data_root      :', cfg.data_root)
print('classes        :', cfg.CLASSES)
print('filter_empty_gt:', cfg.train_dataloader.dataset.dataset.filter_cfg)

# ── 資料載入 ──────────────────────────────────────────────
train_dl = Runner.build_dataloader(cfg.train_dataloader)
val_dl = Runner.build_dataloader(cfg.val_dataloader)
inner = train_dl.dataset.dataset  # ClassBalancedDataset -> CocoDataset
print('train CocoDataset 幀數:', len(inner), '(含 negative)')
print('train ClassBalanced 展開後:', len(train_dl.dataset))
print('valid 幀數:', len(val_dl.dataset))

batch = next(iter(train_dl))

# ── 建模型 + forward(loss) ────────────────────────────────
model = MODELS.build(cfg.model).cuda()
model.train()
data = model.data_preprocessor(batch, True)
print('輸入張量 shape:', tuple(data['inputs'].shape), '(應為 [1,3,H,W])')
assert data['inputs'].shape[1] == 3, '通道數不是 3!'
losses = model(**data, mode='loss')
finite = all(torch.isfinite(v).all() for v in losses.values() if torch.is_tensor(v))
print('loss 數量:', len(losses), '| keys:', list(losses.keys())[:4], '...')
print('所有 loss 有限:', finite)
print('DRY-RUN OK ✓' if finite else 'DRY-RUN 有 NaN/Inf ✗')
