# Co-DINO 5scale + R50 LSJ 3x ── 第 11 次實驗：v13 + Background Subtraction 4ch
# litter-only（單類別）。第 4 通道 = KNN changemap（motion）。
# 動態 test（v13）評估「偵測丟棄動作」。對照 baseline = v12 best epoch 17（純 RGB）。
#
# 相對 v12（co_dino_chiayi_v12_litter_only_36e）的改動：
#   1. 資料集 v13（含 train_changemap/valid_changemap/test_changemap 平行灰階）
#   2. 輸入 4 通道：LoadImageAndChangemap + data_preprocessor 4ch（bgr_to_rgb=False）
#   3. backbone in_channels=4；load_from = conv1 已 inflate(平均初始化) 的 4ch 預訓權重
# ─────────────────────────────────────────────────────────────────────────────

custom_imports = dict(
    imports=["projects.CO-DETR.codetr", "hooks.training_md_logger",
             "hooks.load_changemap", "hooks.data_preprocessor_4ch"],
    allow_failed_imports=False,
)

_base_ = [
    "/opt/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_3x_coco.py"
]

data_root = "/workspace/datasets/coco/chiayi_v13_other_team_litter_dynamic_test_frame/"
NUM_CLASSES = 1
CLASSES = ("litter",)

# ── head 改單類別 + backbone 4 通道 + data_preprocessor 4 通道 ───────────────
model = dict(
    backbone=dict(frozen_stages=1, in_channels=4),
    data_preprocessor=dict(
        type="Det4ChDataPreprocessor",             # 自訂，支援 4 通道 mean/std
        mean=[123.675, 116.28, 103.53, 0.0],       # 第4通道(changemap) mean=0
        std=[58.395, 57.12, 57.375, 128.0],        # 255→~2.0，與 RGB 量級相當
        bgr_to_rgb=False,                          # transform 已以 RGB 載入
        pad_mask=False,
        batch_augments=None,
    ),
    query_head=dict(num_classes=NUM_CLASSES),
    roi_head=[
        dict(
            type="CoStandardRoIHead",
            bbox_roi_extractor=dict(
                type="SingleRoIExtractor",
                roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
                out_channels=256, featmap_strides=[4, 8, 16, 32, 64], finest_scale=56,
            ),
            bbox_head=dict(
                type="Shared2FCBBoxHead",
                in_channels=256, fc_out_channels=1024, roi_feat_size=7,
                num_classes=NUM_CLASSES,
                bbox_coder=dict(type="DeltaXYWHBBoxCoder",
                                target_means=[0.0, 0.0, 0.0, 0.0],
                                target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False, reg_decoded_bbox=True,
                loss_cls=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=12.0),
                loss_bbox=dict(type="GIoULoss", loss_weight=120.0),
            ),
        )
    ],
    bbox_head=[
        dict(
            type="CoATSSHead", num_classes=NUM_CLASSES, in_channels=256,
            stacked_convs=1, feat_channels=256,
            anchor_generator=dict(type="AnchorGenerator", ratios=[1.0],
                                  octave_base_scale=8, scales_per_octave=1,
                                  strides=[4, 8, 16, 32, 64, 128]),
            bbox_coder=dict(type="DeltaXYWHBBoxCoder",
                            target_means=[0.0, 0.0, 0.0, 0.0],
                            target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=12.0),
            loss_bbox=dict(type="GIoULoss", loss_weight=24.0),
            loss_centerness=dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=12.0),
        )
    ],
)

# ── LSJ pipeline（4 通道：LoadImageAndChangemap + 4-tuple pad_val）────────────
image_size = (1024, 1024)
PAD4 = (114, 114, 114, 0)   # RGB 灰 + changemap 0

load_pipeline = [
    dict(type="LoadImageAndChangemap"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=False),
    dict(type="RandomResize", scale=image_size, ratio_range=(0.1, 2.0), keep_ratio=True),
    dict(type="RandomCrop", crop_type="absolute_range", crop_size=image_size,
         recompute_bbox=True, allow_negative_crop=True),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type="RandomFlip", prob=0.5),
    dict(type="Pad", size=image_size, pad_val=dict(img=PAD4)),
    dict(type="PackDetInputs"),
]

train_dataloader = dict(
    batch_size=2, num_workers=4, persistent_workers=True,
    dataset=dict(
        _delete_=True, type="CocoDataset", data_root=data_root,
        metainfo=dict(classes=CLASSES),
        ann_file="annotations/train.json", data_prefix=dict(img="train/"),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),   # 保留背景負樣本
        pipeline=load_pipeline,
    ),
)

test_pipeline = [
    dict(type="LoadImageAndChangemap"),
    dict(type="Resize", scale=image_size, keep_ratio=True),
    dict(type="Pad", size=image_size, pad_val=dict(img=PAD4)),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=False),
    dict(type="PackDetInputs",
         meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")),
]

val_dataloader = dict(
    batch_size=1, num_workers=2,
    dataset=dict(
        _delete_=True, type="CocoDataset", data_root=data_root,
        metainfo=dict(classes=CLASSES),
        ann_file="annotations/valid.json", data_prefix=dict(img="valid/"),
        pipeline=test_pipeline,
    ),
)
val_evaluator = dict(type="CocoMetric", ann_file=data_root + "annotations/valid.json",
                     metric="bbox", classwise=True)

# 動態 test（v13，166 張）
test_dataloader = dict(
    batch_size=1, num_workers=2,
    dataset=dict(
        _delete_=True, type="CocoDataset", data_root=data_root,
        metainfo=dict(classes=CLASSES),
        ann_file="annotations/test.json", data_prefix=dict(img="test/"),
        pipeline=test_pipeline,
    ),
)
test_evaluator = dict(type="CocoMetric", ann_file=data_root + "annotations/test.json",
                      metric="bbox", classwise=True)

# ── 訓練排程（36 epoch，沿用第 5/v12）──────────────────────────────────────────
max_epochs = 36
train_cfg = dict(max_epochs=max_epochs)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type="MultiStepLR", begin=0, end=max_epochs, by_epoch=True, milestones=[24, 33], gamma=0.1),
]
optim_wrapper = dict(
    _delete_=True, type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1)}),
)

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=5,
                    save_best="coco/bbox_mAP", rule="greater"),
)
custom_hooks = [
    dict(type="TrainingMarkdownLogger",
         log_file="/workspace/hdd2_storage/11th_expt_v13_bgs_4ch_litter_36e/training_log.md",
         priority="VERY_LOW"),
]

# ── 4ch 預訓權重（conv1 已 inflate 平均初始化），忽略 head ──────────────────────
load_from = "/workspace/project/models/co_dino_5scale_lsj_r50_3x_coco_4ch_meaninit.pth"
load_from_excluding = ["query_head", "roi_head", "bbox_head"]

work_dir = "/workspace/hdd2_storage/11th_expt_v13_bgs_4ch_litter_36e"
