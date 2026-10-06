# Co-DINO 5scale + R50 LSJ 3x ── 第十三次實驗：v14 動態版（我們自己資料的動態子集）
# 資料集：chiayi_v14_own_dynamic_litter
#   由 v11 過濾而來：litter(2)+garbage(5) 各自用「相鄰幀 GT 位移 >=20px/幀」判定動態種子，
#   再把同 clip 連續影格納入：頭段第一個 seed 往前 30 幀、尾段最後一個 seed 往後只留 5 幀
#   （HEAD_WINDOW=30 / TAIL_WINDOW=5，尾段刻意小，避免片尾拖一長串落地後靜止幀）。
#   train 1237 幀(44 clip) / valid 488 幀(14 clip)（含 ~35-40% negative）；test 沿用原始 v11(594)。
#   尾巴 padding 預設 TAIL_WINDOW=5；使用者比較 tail5/tail10 後，指定 2 部 train clip 用 tail=10
#   （TAIL_OVERRIDE）。另 1 部 valid clip（Day_Litter_535…20240704064730_00m12s）演算法整段當動態，
#   使用者指定只取到 frame 13（END_CAP）。
#   另人工補入 4 部弱動態 clip（自動門檻抓不到、動幅小但肉眼確認是丟垃圾）：起訖 frame 見
#   MANUAL_DYNAMIC；train +2、valid +2。心裡明白這 4 部是較難偵測的動態樣本。
# 目的：固定 domain（都是我們自己嘉義影像），單獨檢驗「動態標註風格」在同 domain 內是否可學。
#       對照 baseline = 第九次 v11（同 config，完整靜+動標註）。
# Classes: car, litter, motorcycle, person, garbage (5 classes)
#
# 改動（相對第九次 v11）：
#   1. data_root → chiayi_v14_own_dynamic_litter（動態子集，連續片段擷取）
#   2. filter_empty_gt True→False：保留 negative 填充幀當背景樣本，壓 litter FP
#   3. 訓練策略其餘完全沿用第九次（LR、class_weight、milestones、ClassBalancedDataset、36e）
#
# 注意：train/valid 過濾成動態、test 仍為原始靜止版 → 存在 train動態/test靜止 的語意落差，
#       這是刻意保留，用來量測「動態訓練能否泛化到靜止落地垃圾」，解讀時需對照第九次 v11。
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

data_root = "/workspace/datasets/coco/chiayi_v14_own_dynamic_litter/"
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
            filter_cfg=dict(filter_empty_gt=False, min_size=32),  # v14: 保留 negative 填充幀
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
# query_head / roi_head / bbox_head: 1e-4 × 1.0 = 1e-4（與第四次 v8_dense 相同）
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
        log_file="/workspace/hdd2_storage/13th_expt_v14_own_dynamic_36e/training_log.md",
        priority="VERY_LOW",
    )
]

# ── Co-DINO 3x COCO 預訓練權重，忽略 head 層（隨機初始化）────────────────────
load_from = "/workspace/project/models/co_dino_5scale_lsj_r50_3x_coco-fe5a6829.pth"
load_from_excluding = ["query_head", "roi_head", "bbox_head"]

# ── 輸出路徑 ──────────────────────────────────────────────────────────────────
# 結果直接寫 HDD（容器內 /workspace/hdd2_storage = host /path/to/data）
# 避免占用近滿的主碟；results/ 另有 host 端 symlink 供瀏覽。
work_dir = "/workspace/hdd2_storage/13th_expt_v14_own_dynamic_36e"
