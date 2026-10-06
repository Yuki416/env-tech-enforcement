"""第十五次 v16 動態版 dry-run：確認 config 可載入、v16 資料讀得到、5 類對齊、negative 保留、
**ClassBalancedDataset 未觸發過採樣**、一個 train batch 能 forward(loss) 且無 NaN。不進行實際訓練。

★ 本次特有的關鍵檢查：`len(ClassBalancedDataset) == len(CocoDataset)`
   v16 的 garbage 影像佔比 f=0.0994 剛好低於預設門檻 0.1，若 oversample_thr 沒設成 0.0，
   MMEngine 會把 r=1.0032 用 math.ceil 進位成 2 → 299 張含 garbage 的影像整份複製 →
   憑空多一個 confound（第十三次沒有這件事）。此處直接 assert 擋住。
"""
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmdet.registry import MODELS
import torch

init_default_scope('mmdet')
import projects  # noqa  Co-DETR registry (PYTHONPATH /opt/mmdetection)

CFG = '/workspace/project/configs/co_dino_chiayi_v16_own_dynamic_36e.py'
cfg = Config.fromfile(CFG)
cfg.train_dataloader.batch_size = 1
cfg.val_dataloader.batch_size = 1

print('data_root      :', cfg.data_root)
print('classes        :', cfg.CLASSES)
print('filter_empty_gt:', cfg.train_dataloader.dataset.dataset.filter_cfg)
print('oversample_thr :', cfg.train_dataloader.dataset.oversample_thr)
print('work_dir       :', cfg.work_dir)
print()

# ── 資料載入 ──────────────────────────────────────────────
train_dl = Runner.build_dataloader(cfg.train_dataloader)
val_dl = Runner.build_dataloader(cfg.val_dataloader)
inner = train_dl.dataset.dataset          # ClassBalancedDataset -> CocoDataset
n_inner = len(inner)
n_wrapped = len(train_dl.dataset)
print('train CocoDataset 幀數      :', n_inner, '(含 negative)')
print('train ClassBalanced 展開後  :', n_wrapped)
print('valid 幀數                  :', len(val_dl.dataset))
print()

# ── ★ 關鍵 assert：沒有過採樣 ─────────────────────────────
assert n_wrapped == n_inner, (
    f'❌ ClassBalancedDataset 觸發了過採樣！{n_inner} → {n_wrapped}（多 {n_wrapped-n_inner} 張）。\n'
    f'   oversample_thr 目前是 {cfg.train_dataloader.dataset.oversample_thr}，應設為 0.0。\n'
    f'   若放著不管，本次就不再是「單一變因＝資料規模」的對照（見 config 檔頭★★）。')
print(f'✅ 無過採樣：ClassBalanced 展開後 == CocoDataset（皆 {n_inner}）')

# ── 規模檢查：對照第十三次 ────────────────────────────────
assert n_inner == 3009, f'❌ train 應為 3009 幀（定案版），實得 {n_inner}'
assert len(val_dl.dataset) == 488, f'❌ valid 應為 488 幀（與第十三次相同），實得 {len(val_dl.dataset)}'
print(f'✅ 規模正確：train 3009（第十三次 1237，2.43×）/ valid 488（與第十三次相同）')
print()

batch = next(iter(train_dl))

# ── 建模型 + forward(loss) ────────────────────────────────
model = MODELS.build(cfg.model).cuda()
model.train()
data = model.data_preprocessor(batch, True)
print('輸入張量 shape:', tuple(data['inputs'].shape), '(應為 [1,3,H,W])')
assert data['inputs'].shape[1] == 3, '❌ 通道數不是 3（本次為純 RGB，非第十四次的 4ch）!'
losses = model(**data, mode='loss')
finite = all(torch.isfinite(v).all() for v in losses.values() if torch.is_tensor(v))
print('loss 數量:', len(losses), '| keys:', list(losses.keys())[:4], '...')
print('所有 loss 有限:', finite)
print('DRY-RUN OK ✓' if finite else 'DRY-RUN 有 NaN/Inf ✗')
