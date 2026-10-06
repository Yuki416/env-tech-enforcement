# Co-DINO 5scale + R50 LSJ 3x ── 第 10 次實驗：v12 (other team litter annotation) 36 epoch
# 實驗分支：litter-only（單一 class）
# Classes: litter only (1 class)
# Dataset: chiayi_v12_other_team_litter_annotation
#   - train/valid: other team（=usera/dataset-litter-3ch）litter 標註，train 排除 2 部與 test 重疊影片
#   - test: chiayi_v11 test，litter+garbage 併成 litter
# Base: co_dino_5scale_r50_lsj_8xb2_3x_coco
# Pretrain: co_dino_5scale_lsj_r50_3x_coco-fe5a6829.pth
#
# 相對第 5 次（co_dino_5th_expt_v8_litter_only_36e）的改動：
#   1. 資料集換成 v12（other team litter）
#   2. filter_empty_gt=False —— 保留背景圖（零標註）作 negative sample，降低 FP
#   3. test_dataloader / test_evaluator 改為獨立的 chiayi_v11 test（不再 = val）
#   4. work_dir 寫到容器內已掛載的 HDD 路徑（/workspace/hdd2_storage/...），避免 results symlink 在容器內斷掉
# ─────────────────────────────────────────────────────────────────────────────

custom_imports = dict(
    imports=["projects.CO-DETR.codetr", "hooks.training_md_logger"],
    allow_failed_imports=False,
)

_base_ = [
    "/opt/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_3x_coco.py"
]

# ── 資料集路徑（container 內）────────────────────────────────────────────────
data_root = "/workspace/datasets/coco/chiayi_v12_other_team_litter_annotation/"
NUM_CLASSES = 1
CLASSES = ("litter",)

# ── 修改 head 的類別數 ────────────────────────────────────────────────────────
model = dict(
    query_head=dict(num_classes=NUM_CLASSES),
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
                    type="CrossEntropyLoss", use_sigmoid=False, loss_weight=12.0
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
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=12.0,
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

# ── Train dataloader（保留背景圖：filter_empty_gt=False）─────────────────────
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    dataset=dict(
        _delete_=True,
        type="CocoDataset",
        data_root=data_root,
        metainfo=dict(classes=CLASSES),
        ann_file="annotations/train.json",
        data_prefix=dict(img="train/"),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=load_pipeline,
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

# ── Val dataloader（other team val）──────────────────────────────────────────
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

# ── Test dataloader（chiayi_v11 test，litter+garbage 併成 litter）────────────
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        _delete_=True,
        type="CocoDataset",
        data_root=data_root,
        metainfo=dict(classes=CLASSES),
        ann_file="annotations/test.json",
        data_prefix=dict(img="test/"),
        pipeline=test_pipeline,
    ),
)

test_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "annotations/test.json",
    metric="bbox",
    classwise=True,
)

# ── 訓練排程（36 epoch，LR drop at 24 & 33；沿用第 5 次）──────────────────────
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

# ── Optimizer（沿用第 5 次：backbone lr_mult=0.1）────────────────────────────
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1)}),
)

# ── Checkpoint：每 epoch 存一次，保留最近 5 個，存最佳 bbox_mAP（單類別=litter）─
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=5,
        save_best="coco/bbox_mAP",
        rule="greater",
    ),
)

# ── 詳細訓練記錄 hook（寫到掛載的 HDD 路徑）──────────────────────────────────
custom_hooks = [
    dict(
        type="TrainingMarkdownLogger",
        log_file="/workspace/hdd2_storage/10th_expt_v12_other_team_litter_36e/training_log.md",
        priority="VERY_LOW",
    )
]

# ── Co-DINO 3x COCO 預訓練權重 fine-tune，忽略 shape 不符的 head 層 ──────────
load_from = "/workspace/project/models/co_dino_5scale_lsj_r50_3x_coco-fe5a6829.pth"
load_from_excluding = ["query_head", "roi_head", "bbox_head"]

# ── 輸出路徑（容器內已掛載的 HDD 路徑，避免 results symlink 在容器內斷掉）─────
work_dir = "/workspace/hdd2_storage/10th_expt_v12_other_team_litter_36e"
