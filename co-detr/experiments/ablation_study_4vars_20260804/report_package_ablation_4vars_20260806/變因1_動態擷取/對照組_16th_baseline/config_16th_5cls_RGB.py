# Co-DINO 5scale + R50 LSJ 3x ── 第十六次實驗：v17 動態版（HEAD=10 + 逐 clip 手動微調）
# 資料集：chiayi_v17_own_dynamic_litter_head10_manual
#   由 v17(122部) 過濾而來。配方沿用第十五次，**唯一實質變動 = 資料集**：
#     (1) v16→v17：首建重標 7 部 + R2 修正輪重標 4 部 + 新增 1 部 valid garbage。
#     (2) 動態擷取 HEAD_WINDOW 30→10（前緣收緊、砍靜止填充）。
#     (3) 8 部 clip 逐一手動釘死擷取範圍（RANGE_OVERRIDE，絕對頭尾幀）。
#   train 2494 幀(99 clip) / valid 373 幀(15 clip)（含 negative 填充幀）；
#   test 用 v17 完整版 655 幀(8 clip)，不過濾。
#
# 目的：承接第十五次（v16、HEAD=30，效果不佳）。假設病灶在資料前處理（早期標註品質 + 動態範圍太鬆）。
#
# ★ 主判定 baseline = 第十三次 v14 的 test_static594（594 張/1823 框，同一把尺）：litter AP@0.5 = 0.254。
#   ⚠️ 相對第十五次，本次同時動了「資料集(v16→v17) + HEAD(30→10) + 8 部手動」→ **非單一變因**。
#      跨次乾淨比較只看 test_static594（獨立 test、與 13/15 次同尺）。
#
# 改動（相對第十五次 v16）：
#   1. data_root → chiayi_v17_own_dynamic_litter_head10_manual
#   2. work_dir / training_log 路徑 → 16th_expt_...
#   3. oversample_thr 維持 0.0（見下方★★；garbage 影像佔比又變，0.0 對任何分布都保證不過採樣）
#   其餘（LR、class_weight、milestones、36e、filter_empty_gt=False、save_best、load_from）完全沿用。
#
# ★★ oversample_thr=0.0（沿用第十五次的理由，本次同樣重要）：
#   MMEngine ClassBalancedDataset：r(c)=max(1,sqrt(thr/f(c)))，再對每張圖取 math.ceil(r)。
#   garbage 影像佔比 f 換資料集就變（本次 train 2494 幀），可能剛好跨過門檻 0.1 被 ceil 進位成 2×。
#   設 thr=0.0 → r=max(1.0,0)=1.0 → ceil=1，**對任何分布都保證不過採樣**，忠實重現第十三/十五次的實際行為。
#   dryrun 的 `len(ClassBalancedDataset)==len(CocoDataset)` assert 直接擋住回歸。
#
# ★★★ ckpt 機制與 valid（重要，報告需說明）：
#   save_best=coco/litter_precision（valid 上 litter 類 COCO AP）、rule=greater —— **機制與第十五次逐字相同**。
#   但 **valid 這把尺本身變了**：15th valid=488幀/14clips(全 v11 舊)；16th valid=373幀/15clips
#   （含新增 00m19s、R2 重標 02m13s、手動調 00m12s）。
#   → 挑「最佳 ckpt」是用不同 valid 判定的 → **valid 絕對分數不可跨次(16th vs 15th)直接比**；
#      但 16th valid 較貼近午後/新 domain，選 ckpt 反而更有代表性。主判定 test_static594 不受此影響。
#
# 注意：train/valid 為動態子集、test 為完整版（含靜止落地垃圾）→ 語意落差刻意保留。
#       v17 完整 655 test 含已知同片洩漏（見實驗夾 README），引用時須標註虛高；主判定用 test_static594。
#
# LR decay 路徑：
#   ep1-27    backbone 1e-6  / head 1e-4
#   ep28-33   backbone 1e-7  / head 1e-5  (×0.1)
#   ep34-36   backbone 1e-8  / head 1e-6  (×0.1)
# ─────────────────────────────────────────────────────────────────────────────

custom_imports = dict(
    imports=[
        "projects.CO-DETR.codetr",
        "hooks.training_md_logger",
        "hooks.weighted_focal_loss",
    ],
    allow_failed_imports=False,
)

_base_ = [
    "/opt/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_3x_coco.py"
]

data_root = "/workspace/datasets/coco/chiayi_v17_own_dynamic_litter_head10_manual/"
NUM_CLASSES = 5
CLASSES = ("car", "litter", "motorcycle", "person", "garbage")

# FocalLoss / QualityFocalLoss（sigmoid multi-label，C=5）：長度 5
CLASS_WEIGHT = [1.0, 3.0, 1.0, 1.0, 3.0]
# CrossEntropyLoss（softmax，C=num_classes+1=6，含 background）：長度 6
CLASS_WEIGHT_CE = [1.0, 3.0, 1.0, 1.0, 3.0, 1.0]  # background weight=1.0

# ── 修改 head 的類別數與 loss 設定 ────────────────────────────────────────────
model = dict(
    query_head=dict(
        num_classes=NUM_CLASSES,
        loss_cls=dict(
            type="ClassWeightedQualityFocalLoss",
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0,
            class_weight=CLASS_WEIGHT,
        ),
    ),
    roi_head=[
        dict(
            type="CoStandardRoIHead",
            bbox_roi_extractor=dict(
                type="SingleRoIExtractor",
                roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64],
                finest_scale=56,
            ),
            bbox_head=dict(
                type="Shared2FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=NUM_CLASSES,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type="CrossEntropyLoss",
                    use_sigmoid=False,
                    loss_weight=12.0,
                    class_weight=CLASS_WEIGHT_CE,  # 長度 6（含 background）
                ),
                loss_bbox=dict(type="GIoULoss", loss_weight=120.0),
            ),
        )
    ],
    bbox_head=[
        dict(
            type="CoATSSHead",
            num_classes=NUM_CLASSES,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type="AnchorGenerator",
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[4, 8, 16, 32, 64, 128],
            ),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            loss_cls=dict(
                type="ClassWeightedFocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=12.0,
                class_weight=CLASS_WEIGHT,
            ),
            loss_bbox=dict(type="GIoULoss", loss_weight=24.0),
            loss_centerness=dict(
                type="CrossEntropyLoss", use_sigmoid=True, loss_weight=12.0
            ),
        )
    ],
    backbone=dict(frozen_stages=1),
    data_preprocessor=dict(
        pad_mask=False,
        batch_augments=None,
    ),
)

# ── LSJ pipeline（停用 CopyPaste）────────────────────────────────────────────
image_size = (1024, 1024)

load_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=False),
    dict(
        type="RandomResize",
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True,
    ),
    dict(
        type="RandomCrop",
        crop_type="absolute_range",
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True,
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Pad", size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type="PackDetInputs"),
]

# ── Train dataloader（ClassBalancedDataset + persistent_workers）─────────────
# oversample_thr=0.0 → 保證不過採樣（理由見檔頭★★）。保留 wrapper 以與第十三/十五次維持最小結構差異。
# persistent_workers=True 必須保留：epoch boundary deadlock 的解法。
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        _delete_=True,
        type="ClassBalancedDataset",
        oversample_thr=0.0,
        dataset=dict(
            type="CocoDataset",
            data_root=data_root,
            metainfo=dict(classes=CLASSES),
            ann_file="annotations/train.json",
            data_prefix=dict(img="train/"),
            filter_cfg=dict(filter_empty_gt=False, min_size=32),  # 保留 negative 填充幀
            pipeline=load_pipeline,
        ),
    ),
)

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", scale=image_size, keep_ratio=True),
    dict(type="Pad", size=image_size, pad_val=dict(img=(114, 114, 114))),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=False),
    dict(
        type="PackDetInputs",
        meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor"),
    ),
]

# ── Val / Test dataloader ─────────────────────────────────────────────────────
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        _delete_=True,
        type="CocoDataset",
        data_root=data_root,
        metainfo=dict(classes=CLASSES),
        ann_file="annotations/valid.json",
        data_prefix=dict(img="valid/"),
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/valid.json",
    metric="bbox",
    classwise=True,
)

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# ── 訓練排程（36 epoch，LR drop at 28 & 34）──────────────────────────────────
max_epochs = 36
train_cfg = dict(max_epochs=max_epochs)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[28, 34],
        gamma=0.1,
    ),
]

# ── 差異化 LR ─────────────────────────────────────────────────────────────────
# backbone: 1e-4 × 0.01 = 1e-6（強力保護預訓練特徵）
# query_head / roi_head / bbox_head: 1e-4 × 1.0 = 1e-4
# ⚠️ 刻意維持 0.01，不採第十四次的 0.1 —— 第十四次因此多了一個混淆變因。
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.01),
            "query_head": dict(lr_mult=1.0),
            "roi_head": dict(lr_mult=1.0),
            "bbox_head": dict(lr_mult=1.0),
        }
    ),
)

# ── Checkpoint：監控 litter precision ────────────────────────────────────────
# 機制與第十五次逐字相同（save_best=coco/litter_precision, greater）。
# ⚠️ valid 這把尺變了（373幀/15clips，含新增+R2重標+手動調）→ 選出的 best ckpt 與 15th 不可直接比 valid 分數。
#    詳見檔頭★★★。主判定 test_static594 不受影響。
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=5,
        save_best="coco/litter_precision",
        rule="greater",
    ),
)

# ── 詳細訓練記錄 hook ─────────────────────────────────────────────────────────
custom_hooks = [
    dict(
        type="TrainingMarkdownLogger",
        log_file="/workspace/hdd2_storage/16th_expt_v17_own_dynamic_122vids_36e/training_log.md",
        priority="VERY_LOW",
    )
]

# ── Co-DINO 3x COCO 預訓練權重，忽略 head 層（隨機初始化）────────────────────
load_from = "/workspace/project/models/co_dino_5scale_lsj_r50_3x_coco-fe5a6829.pth"
load_from_excluding = ["query_head", "roi_head", "bbox_head"]

# ── 輸出路徑 ──────────────────────────────────────────────────────────────────
# 結果直接寫 HDD（容器內 /workspace/hdd2_storage = host /path/to/data）
# 避免占用近滿的主碟；results/ 另有 host 端 symlink 供瀏覽。
work_dir = "/workspace/hdd2_storage/16th_expt_v17_own_dynamic_122vids_36e"
