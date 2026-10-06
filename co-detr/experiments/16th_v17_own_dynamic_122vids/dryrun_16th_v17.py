"""第十六次 v17 動態版（HEAD=10 + 手動微調）dry-run：確認 config 可載入、資料讀得到、5 類對齊、
negative 保留、**ClassBalancedDataset 未觸發過採樣**、一個 train batch 能 forward(loss) 且無 NaN。不實際訓練。

★ 關鍵檢查（沿用第十五次）：`len(ClassBalancedDataset) == len(CocoDataset)`
   garbage 影像佔比換資料集就變，可能剛好跨過預設門檻 0.1 被 math.ceil 進位成 2× → 憑空多一個 confound。
   oversample_thr=0.0 對任何分布都保證不過採樣；此處直接 assert 擋住並印出實際 garbage 影像佔比供記錄。

★ 規模檢查改為本次定案值：train 2494 / valid 373（見實驗夾 README / final_check_manual.py）。

用法（docker 內）：PYTHONUTF8=1 python3 /workspace/project/experiments/16th_v17_own_dynamic_122vids/dryrun_16th_v17.py
"""
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmdet.registry import MODELS
import torch

init_default_scope('mmdet')
import projects  # noqa  Co-DETR registry (PYTHONPATH /opt/mmdetection)

CFG = '/workspace/project/configs/co_dino_chiayi_v17_own_dynamic_head10_manual_36e.py'
cfg = Config.fromfile(CFG)
cfg.train_dataloader.batch_size = 1
cfg.val_dataloader.batch_size = 1

print('data_root      :', cfg.data_root)
print('classes        :', cfg.CLASSES)
print('filter_empty_gt:', cfg.train_dataloader.dataset.dataset.filter_cfg)
print('oversample_thr :', cfg.train_dataloader.dataset.oversample_thr)
print('work_dir       :', cfg.work_dir)
print('save_best      :', cfg.default_hooks.checkpoint.save_best)
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

# garbage 影像佔比（供記錄；ClassBalancedDataset 過採樣觸發條件的輸入）。
# 純記錄用，算不到不影響主流程（守門是下方「無過採樣」assert）。garbage=category_id 5 → 0-based label 4。
try:
    gar_imgs = sum(
        1 for i in range(n_inner)
        if any(inst.get('bbox_label') == 4 for inst in inner.get_data_info(i).get('instances', []))
    )
    print(f'garbage 影像佔比 f          : {gar_imgs}/{n_inner} = {gar_imgs/n_inner:.4f}')
except Exception as e:
    print(f'garbage 影像佔比            : （略過計算：{e}）')
print()

# ── ★ 關鍵 assert：沒有過採樣 ─────────────────────────────
assert n_wrapped == n_inner, (
    f'❌ ClassBalancedDataset 觸發了過採樣！{n_inner} → {n_wrapped}（多 {n_wrapped-n_inner} 張）。\n'
    f'   oversample_thr 目前是 {cfg.train_dataloader.dataset.oversample_thr}，應設為 0.0（見 config 檔頭★★）。')
print(f'✅ 無過採樣：ClassBalanced 展開後 == CocoDataset（皆 {n_inner}）')

# ── 規模檢查：本次定案值 ──────────────────────────────────
assert n_inner == 2494, f'❌ train 應為 2494 幀（手動微調定案版），實得 {n_inner}'
assert len(val_dl.dataset) == 373, f'❌ valid 應為 373 幀（手動微調定案版），實得 {len(val_dl.dataset)}'
print(f'✅ 規模正確：train 2494 / valid 373（第十六次手動微調定案版）')
print()

batch = next(iter(train_dl))

# ── 建模型 + forward(loss) ────────────────────────────────
model = MODELS.build(cfg.model).cuda()
model.train()
data = model.data_preprocessor(batch, True)
print('輸入張量 shape:', tuple(data['inputs'].shape), '(應為 [1,3,H,W])')
assert data['inputs'].shape[1] == 3, '❌ 通道數不是 3（本次為純 RGB）!'
losses = model(**data, mode='loss')
finite = all(torch.isfinite(v).all() for v in losses.values() if torch.is_tensor(v))
print('loss 數量:', len(losses), '| keys:', list(losses.keys())[:4], '...')
print('所有 loss 有限:', finite)
print('DRY-RUN OK ✓' if finite else 'DRY-RUN 有 NaN/Inf ✗')
