# Co-DINO 5scale + R50 LSJ 3x ── 第七次實驗：Loss Compensate 策略
# 資料集：chiayi_v9_garbage_litter（與第六次相同）
# Classes: car, litter, motorcycle, person, garbage (5 classes)
#
# 改動（相對第六次 v9）：
#   1. 差異化 LR：backbone 1.25e-6，head(query/roi/bbox) 1e-4（lr_mult=8.0）
#   2. class_weight：三個 head 全部套用
#      query_head → ClassWeightedQualityFocalLoss（class dim sum 前插入，自訂）[1,5,1,1,5]
#      roi_head   → CrossEntropyLoss + class_weight（原生支援，需含 background 共 6 元素）[1,5,1,1,5,1]
#      bbox_head  → ClassWeightedFocalLoss（自訂 wrapper）[1,5,1,1,5]
#   3. ClassBalancedDataset：oversample_thr=0.1，過採樣含 litter/garbage 的圖片
#   4. persistent_workers=True：防止 epoch boundary DataLoader 死鎖（第六次 ep13 曾發生）
#   5. LR milestones 改回 [24, 33]（v9 用 [28,34] 效果不佳）
#   6. save_best 改為 coco/litter_precision
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

data_root = "/workspace/datasets/coco/chiayi_v9_garbage_litter/"
NUM_CLASSES = 5
CLASSES = ("car", "litter", "motorcycle", "person", "garbage")

# FocalLoss / QualityFocalLoss（sigmoid multi-label，C=5）：長度 5
CLASS_WEIGHT = [1.0, 5.0, 1.0, 1.0, 5.0]
# CrossEntropyLoss（softmax，C=num_classes+1=6，含 background）：長度 6
CLASS_WEIGHT_CE = [1.0, 5.0, 1.0, 1.0, 5.0, 1.0]  # background weight=1.0

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
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        _delete_=True,
        type="ClassBalancedDataset",
        oversample_thr=0.1,
        dataset=dict(
            type="CocoDataset",
            data_root=data_root,
            metainfo=dict(classes=CLASSES),
            ann_file="annotations/train.json",
            data_prefix=dict(img="train/"),
            filter_cfg=dict(filter_empty_gt=True, min_size=32),
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

# ── 訓練排程（36 epoch，LR drop at 24 & 33）──────────────────────────────────
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
        milestones=[24, 33],
        gamma=0.1,
    ),
]

# ── 差異化 LR ─────────────────────────────────────────────────────────────────
# backbone: 1.25e-5 × 0.1 = 1.25e-6（保護預訓練特徵）
# query_head / roi_head / bbox_head: 1.25e-5 × 8.0 = 1e-4（加速隨機初始化 head 學習）
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1.25e-5, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(lr_mult=0.1),
            "query_head": dict(lr_mult=8.0),
            "roi_head": dict(lr_mult=8.0),
            "bbox_head": dict(lr_mult=8.0),
        }
    ),
)

# ── Checkpoint：監控 litter AP ────────────────────────────────────────────────
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
        log_file="/workspace/project/results/7th_expt_v9_loss_compensate_36e/training_log.md",
        priority="VERY_LOW",
    )
]

# ── Co-DINO 3x COCO 預訓練權重，忽略 head 層（隨機初始化）────────────────────
load_from = "/workspace/project/models/co_dino_5scale_lsj_r50_3x_coco-fe5a6829.pth"
load_from_excluding = ["query_head", "roi_head", "bbox_head"]

# ── 輸出路徑 ──────────────────────────────────────────────────────────────────
work_dir = "/workspace/project/results/7th_expt_v9_loss_compensate_36e"
