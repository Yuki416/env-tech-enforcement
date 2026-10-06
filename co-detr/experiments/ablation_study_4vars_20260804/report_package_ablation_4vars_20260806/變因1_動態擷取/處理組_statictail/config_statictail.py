# Co-DINO 5scale + R50 LSJ 3x ── 消融實驗 變因①：動態擷取（靜止尾 statictail）處理組
# 資料集：chiayi_v17_own_dynamic_head10_manual_statictail_ablation
#   = 第 16 次動態子集（2494 幀）的**嚴格超集**：每個 clip 從 16th 選幀終點往後、延伸到該 clip
#     最後一個 litter/garbage 有標幀，延伸段每 3 幀取 1 張抽稀 → +1057 幀（train 3551 幀 / 99 clip）。
#   valid 373 幀（= 16th，逐檔相同）；test 655 幀（+ test_static594 / dynamic 8v,7v 子集）。
#   詳見 experiments/ablation_study_4vars_20260804/變因1_動態擷取_實驗記錄.md。
#
# 對照組 = 第 16 次（chiayi_v17_own_dynamic_litter_head10_manual，已訓練，best=ep23）。
# 自變因 = 訓練集是否納入「垃圾落地/動作結束後仍有標註的靜止尾幀」。
# 依變因 = test static litter AP@0.5（主看 test_static594 乾淨版）。
#
# ★ 相對第 16 次，只有 3 個刻意改動：
#   (1) data_root → statictail_ablation（＝加入靜止尾，唯一變因）。
#   (2) max_keep_ckpts 5 → -1（**全留 36 顆**）：為畫「全 36 epoch 的 test-static AP 曲線」做診斷，
#       檢查靜止尾潛力有沒有被純動態 valid 的選擇機制埋掉（見實驗記錄第 6 節）。
#   (3) work_dir / training_log 路徑 → ablation_var1_statictail_36e。
#   其餘（LR、class_weight、milestones、36e、filter_empty_gt=False、oversample_thr=0.0、
#         save_best=coco/litter_precision、load_from、pipeline）**全部逐字沿用第 16 次**。
#
# ★★ valid 維持與 16th 相同（373 幀）＝ **程序控制（procedure control）**：
#   valid 不是量測工具（依變因 test-static AP 不經過 valid），而是模型選擇程序的一部分。
#   固定它的理由：train 與選擇程序若同時變，test 差異就無法歸因給 train。故只動 train、選擇程序全同。
#   ⚠️ 已知風險：ckpt 用純動態 valid 選，可能系統性偏向「動態好、靜止未必好」那顆 → 偏誤方向與
#      預先登記的「test static 會失敗」預測同向。故評估採雙層（best 對 best + 同 epoch 對點 + 全曲線診斷）。
#
# ★★ oversample_thr=0.0（沿用）：ClassBalancedDataset r(c)=max(1,sqrt(thr/f(c)))，thr=0 → r=1 → 不過採樣。
#   dryrun 的 len(ClassBalancedDataset)==len(CocoDataset) assert 擋回歸。
#
# LR decay 路徑（同 16th）：
#   ep1-27  backbone 1e-6 / head 1e-4 ; ep28-33 ×0.1 ; ep34-36 ×0.1
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

data_root = "/workspace/datasets/coco/chiayi_v17_own_dynamic_head10_manual_statictail_ablation/"
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
# oversample_thr=0.0 → 保證不過採樣（理由見檔頭★★）。
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

# ── 差異化 LR（同 16th）───────────────────────────────────────────────────────
# backbone: 1e-4 × 0.01 = 1e-6；head: 1e-4 × 1.0 = 1e-4。刻意維持 0.01（不採 14th 的 0.1）。
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
# 機制與 16th 逐字相同（save_best=coco/litter_precision, greater），
# ★唯一差異：max_keep_ckpts=-1（全留 36 顆），為畫完整 test-static 曲線（實驗記錄第 6 節）。
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=-1,
        save_best="coco/litter_precision",
        rule="greater",
    ),
)

# ── 詳細訓練記錄 hook ─────────────────────────────────────────────────────────
custom_hooks = [
    dict(
        type="TrainingMarkdownLogger",
        log_file="/workspace/hdd2_storage/ablation_var1_statictail_36e/training_log.md",
        priority="VERY_LOW",
    )
]

# ── Co-DINO 3x COCO 預訓練權重，忽略 head 層（隨機初始化）────────────────────
load_from = "/workspace/project/models/co_dino_5scale_lsj_r50_3x_coco-fe5a6829.pth"
load_from_excluding = ["query_head", "roi_head", "bbox_head"]

# ── 輸出路徑 ──────────────────────────────────────────────────────────────────
work_dir = "/workspace/hdd2_storage/ablation_var1_statictail_36e"
