# Co-DINO 5scale + R50 LSJ 3x ── 第 17 次實驗：v17 動態版，只留 litter + garbage 兩類
# 資料集：chiayi_v17_own_dynamic_litter_head10_manual（= 第16次同一份 V17，不另產新檔）
#   本次「唯一資料面差異」= 只取 litter/garbage 兩類（metainfo 子集），其餘 car/person/motorcycle
#   標註由 MMDet 自動丟棄、category id 重映（litter→0, garbage→1，依名稱對應，與 json id 順序無關）。
#   → 88% 標註（car 5207 + motorcycle 4380 + person 3619）消失，剩 litter 1410 + garbage 391。
#   → 918/2494 train 圖變純 negative（原本只有車/人/機車），依慣例保留當 negative（filter_empty_gt=False）。
#
# ★ 對照方法 = 第 10 次（litter-only 單類別分支）：**完全照第 10 次配方**
#     plain FocalLoss / CrossEntropyLoss（無 class weighting，不 import weighted_focal_loss hook）；
#     backbone lr_mult=0.1；milestones=[24,33]；**不套 ClassBalancedDataset wrapper**（plain CocoDataset）。
#   與第 10 次唯一差別：(a) 資料集 = 自己的 v17 動態版（非 other-team v12）；
#                       (b) 兩類（litter+garbage）而非單類 litter；
#                       (c) save_best 用 litter_precision（見下），非第 10 次的 bbox_mAP。
#
# ★★ 假設：第 16 次（5 類、v17）litter AP 崩到 0.041。本次移除 car/person/motorcycle 三個干擾類別，
#     測「把模型聚焦在垃圾類別」是否救回 litter/garbage 偵測（第 10 次的聚焦法在自己資料上是否有效）。
#     ⚠️ 相對第 16 次同時動了「class set(5→2) + 配方(改回第10次 lr_mult0.1/milestones[24,33]/去掉加權與ClassBalanced)」
#        → 對第 16 次而言非單一變因；本次是「對照第 10 次方法」的乾淨重現，不宣稱對第 16 次單一變因。
#
# ★★★ 主判定 baseline 不變 = test_static594 的 litter AP@0.5：第13次 v14=0.254、第16次 v17(5cls)=0.041。
#     （訓練 config 的 test_dataloader=val，真正主判定用獨立 eval 腳本在 test_static594 上跑，與第16次一致。）
#
# LR decay 路徑（backbone lr_mult=0.1 → 1e-5；head 1e-4；milestones=[24,33]）：
#   ep1-24    backbone 1e-5  / head 1e-4
#   ep25-33   backbone 1e-6  / head 1e-5  (×0.1)
#   ep34-36   backbone 1e-7  / head 1e-6  (×0.1)
# ─────────────────────────────────────────────────────────────────────────────

custom_imports = dict(
    imports=["projects.CO-DETR.codetr", "hooks.training_md_logger"],
    allow_failed_imports=False,
)

_base_ = [
    "/opt/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_3x_coco.py"
]

# ── 資料集路徑（container 內；= 第16次同一份 V17）────────────────────────────
data_root = "/workspace/datasets/coco/chiayi_v17_own_dynamic_litter_head10_manual/"
NUM_CLASSES = 2
CLASSES = ("litter", "garbage")  # 依名稱對應 json，litter→0, garbage→1

# ── 修改 head 的類別數（plain loss，無 class weighting；沿用第 10 次）──────────
model = dict(
    query_head=dict(num_classes=NUM_CLASSES),  # loss_cls 沿用 base（QualityFocalLoss, sigmoid）
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

# ── Train dataloader（plain CocoDataset，無 ClassBalanced；保留背景圖）────────
# ★ 不套 ClassBalancedDataset（照第 10 次）。filter_empty_gt=False → 918 張純 negative 保留。
# persistent_workers=True 必須保留：epoch boundary deadlock 的解法。
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

# ── Val dataloader（v17 valid）──────────────────────────────────────────────
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
    classwise=True,  # 吐出 coco/litter_precision、coco/garbage_precision 供 save_best 監控
)

# ── Test dataloader = val（真正主判定用獨立 eval 腳本跑 test_static594，與第16次一致）─
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# ── 訓練排程（36 epoch，LR drop at 24 & 33；照第 10 次）──────────────────────
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

# ── Optimizer（照第 10 次：backbone lr_mult=0.1）─────────────────────────────
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1)}),
)

# ── Checkpoint：save_best=litter_precision（litter 為主，覆蓋第10次的 bbox_mAP）─
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=5,
        save_best="coco/litter_precision",
        rule="greater",
    ),
)

# ── 詳細訓練記錄 hook（寫到掛載的 HDD 路徑）──────────────────────────────────
custom_hooks = [
    dict(
        type="TrainingMarkdownLogger",
        log_file="/workspace/hdd2_storage/17th_expt_v17_own_dynamic_2cls_36e/training_log.md",
        priority="VERY_LOW",
    )
]

# ── Co-DINO 3x COCO 預訓練權重 fine-tune，忽略 shape 不符的 head 層 ──────────
load_from = "/workspace/project/models/co_dino_5scale_lsj_r50_3x_coco-fe5a6829.pth"
load_from_excluding = ["query_head", "roi_head", "bbox_head"]

# ── 輸出路徑（容器內已掛載的 HDD 路徑）──────────────────────────────────────
work_dir = "/workspace/hdd2_storage/17th_expt_v17_own_dynamic_2cls_36e"
