# Co-DINO 5scale + R50 LSJ 3x ── 第 12 次實驗：v13 + Background Subtraction 4ch
# litter-only（單類別）。第 4 通道 = KNN changemap（motion）。
# 資料集與第 11 次「完全相同」(v13 dynamic test)。本次只改「微調/凍結策略」。
#
# 相對第 11 次（co_dino_chiayi_v13_4ch_litter_36e）的改動 ── 全部針對「讓遮罩通道學得到」：
#   ① frozen_stages 1 → -1   ：解凍 stem(conv1)+layer1，否則第4通道投影永遠停在 init
#   ② paramwise 新增 backbone.conv1 lr_mult=1.0：遮罩通道是新東西，需要比其餘 backbone(0.1x)大的 lr
#   ④ Conv1Ch4NormLogger     ：診斷探針，追蹤 conv1 第4通道 L2 norm 是否真的在動
#   （③ norm_eval 本次決定不開，維持 base 的 True：bs=1 下單張 batch 統計量太吵，
#     保留 COCO 凍結 running stats 較穩；也讓變因縮到只有「凍結策略+conv1 lr」。）
#   其餘（資料集、4ch pipeline、單類別 head、load_from、排程）一律沿用第 11 次。
# ─────────────────────────────────────────────────────────────────────────────

custom_imports = dict(
    imports=["projects.CO-DETR.codetr", "hooks.training_md_logger",
             "hooks.load_changemap", "hooks.data_preprocessor_4ch",
             "hooks.conv1_ch4_norm_logger"],
    allow_failed_imports=False,
)

_base_ = [
    "/opt/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_3x_coco.py"
]

data_root = "/workspace/datasets/coco/chiayi_v13_other_team_litter_dynamic_test_frame/"
NUM_CLASSES = 1
CLASSES = ("litter",)

# ── head 改單類別 + backbone 4 通道（解凍 stem）+ data_preprocessor 4 通道 ───────
model = dict(
    backbone=dict(
        frozen_stages=-1,          # ① 全解凍：conv1(含第4通道)+layer1 都可訓練
        norm_eval=True,            # 維持 base 預設：bs=1 下 BN 用凍結 running stats（不開 ③）
        in_channels=4,
    ),
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
# 純幾何增強，無 PhotoMetricDistortion → 不會污染遮罩通道。RGB+遮罩併成 4ch 後
# 一起經 RandomResize/RandomCrop/RandomFlip/Pad，天然同步。沿用第 11 次。
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

# 動態 test（v13，166 張）── 與第 11 次相同
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

# ── 訓練排程（36 epoch，沿用第 11 次）─────────────────────────────────────────
max_epochs = 36
train_cfg = dict(max_epochs=max_epochs)
param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type="MultiStepLR", begin=0, end=max_epochs, by_epoch=True, milestones=[24, 33], gamma=0.1),
]

# ── 微調策略（本次核心改動）─────────────────────────────────────────────────
# paramwise custom_keys 以「最長匹配」勝出：backbone.conv1.* 命中較長的 "backbone.conv1"
# → conv1（含第4通道）拿 lr_mult=1.0（= base lr，與重置的 head 同級，從頭學）；
#   其餘 backbone 仍是 0.1x 微調。
optim_wrapper = dict(
    _delete_=True, type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.0001, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        "backbone.conv1": dict(lr_mult=1.0),   # ② 遮罩通道所在的第一層，給大 lr
        "backbone": dict(lr_mult=0.1),         # 其餘 backbone 維持微調
    }),
)

default_hooks = dict(
    checkpoint=dict(type="CheckpointHook", interval=1, max_keep_ckpts=5,
                    save_best="coco/bbox_mAP", rule="greater"),
)
custom_hooks = [
    dict(type="TrainingMarkdownLogger",
         log_file="/workspace/hdd2_storage/12th_expt_v13_bgs_4ch_stem_unfreeze_36e/training_log.md",
         priority="VERY_LOW"),
    dict(type="Conv1Ch4NormLogger"),   # ④ 診斷探針：conv1 第4通道 L2 norm
]

# ── 4ch 預訓權重（conv1 已 inflate 平均初始化），忽略 head ── 與第 11 次相同
load_from = "/workspace/project/models/co_dino_5scale_lsj_r50_3x_coco_4ch_meaninit.pth"
load_from_excluding = ["query_head", "roi_head", "bbox_head"]

work_dir = "/workspace/hdd2_storage/12th_expt_v13_bgs_4ch_stem_unfreeze_36e"
