# 第17次 config dryrun：不訓練，只建 model+dataset 驗證關鍵設定
import warnings; warnings.filterwarnings("ignore")
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmdet.registry import DATASETS, MODELS
init_default_scope("mmdet")
cfg = Config.fromfile("/workspace/project/configs/co_dino_chiayi_v17_own_dynamic_2cls_litter_garbage_36e.py")

print("=== train_dataloader.dataset.type ===")
ds_cfg = cfg.train_dataloader.dataset
print("  wrapper type:", ds_cfg.type, "(應為 CocoDataset，不是 ClassBalancedDataset)")
print("  filter_empty_gt:", ds_cfg.filter_cfg.get("filter_empty_gt"))
print("  metainfo classes:", ds_cfg.metainfo["classes"])

ds = DATASETS.build(ds_cfg)
print("\n=== 建好的 train dataset ===")
print("  len(train dataset):", len(ds), "(應=2494：無 ClassBalanced 過採樣、negative 保留)")
print("  dataset metainfo classes:", ds.metainfo["classes"])
# 掃前面若干張確認 label 只有 {0,1}
labels=set()
for i in range(len(ds)):
    info = ds.get_data_info(i)
    for inst in info["instances"]:
        labels.add(inst["bbox_label"])
print("  train 所有 bbox_label 取值:", sorted(labels), "(應為 [0, 1])")
# 統計每類框數 + 純 negative 圖數
import collections
cnt=collections.Counter(); neg=0
for i in range(len(ds)):
    info=ds.get_data_info(i)
    if len(info["instances"])==0: neg+=1
    for inst in info["instances"]: cnt[inst["bbox_label"]]+=1
names=ds.metainfo["classes"]
print("  每類框數:", {names[k]:v for k,v in sorted(cnt.items())})
print("  純 negative 圖數:", neg, "(應≈918)")

print("\n=== model head num_classes ===")
print("  query_head.num_classes:", cfg.model.query_head.num_classes)
print("  roi_head[0].bbox_head.num_classes:", cfg.model.roi_head[0].bbox_head.num_classes)
print("  bbox_head[0].num_classes:", cfg.model.bbox_head[0].num_classes)
print("  query_head loss_cls type:", cfg.model.query_head.get("loss_cls", "inherit-base"))
print("  bbox_head loss_cls type:", cfg.model.bbox_head[0].loss_cls.type, "(應為 FocalLoss，非 ClassWeighted)")
print("  roi_head loss_cls type:", cfg.model.roi_head[0].bbox_head.loss_cls.type)
print("\n=== save_best / optim / milestones ===")
print("  save_best:", cfg.default_hooks.checkpoint.save_best)
print("  backbone lr_mult:", cfg.optim_wrapper.paramwise_cfg.custom_keys.backbone.lr_mult)
print("  milestones:", cfg.param_scheduler[1].milestones)
print("  custom_imports:", cfg.custom_imports.imports)
print("\nDRYRUN_OK")
