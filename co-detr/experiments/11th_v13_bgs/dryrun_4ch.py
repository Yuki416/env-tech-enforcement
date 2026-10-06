"""4ch 管線 dry-run：建模型 + 取一個 train batch + forward(loss)，確認 4 通道流通無誤。"""
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmdet.registry import MODELS
import torch

init_default_scope('mmdet')
import projects  # noqa  (Co-DETR registry via PYTHONPATH /opt/mmdetection)
import hooks.load_changemap          # noqa  註冊 transform
import hooks.data_preprocessor_4ch   # noqa  註冊 4ch preprocessor

cfg = Config.fromfile('/workspace/project/configs/co_dino_chiayi_v13_4ch_litter_36e.py')
cfg.train_dataloader.batch_size = 1   # 與實際訓練一致，避免 OOM

model = MODELS.build(cfg.model).cuda()
model.train()
dl = Runner.build_dataloader(cfg.train_dataloader)
batch = next(iter(dl))
data = model.data_preprocessor(batch, True)
print('輸入張量 shape:', tuple(data['inputs'].shape), '(應為 [B,4,H,W])')
assert data['inputs'].shape[1] == 4, '通道數不是 4!'
losses = model(**data, mode='loss')
finite = all(torch.isfinite(v).all() for v in losses.values() if torch.is_tensor(v))
print('loss 數量:', len(losses), '| 前幾個:', list(losses.keys())[:4])
print('所有 loss 有限:', finite)
print('DRY-RUN OK ✓' if finite else 'DRY-RUN 有 NaN/Inf ✗')
