# Co-DINO 5scale + R50 LSJ 3x ── 第 18 次實驗：v18 = v17 手動動態子集 + Background Subtraction 4ch
# 資料集：chiayi_v18_own_dynamic_bgs_122vids
#   = 第 16 次的 chiayi_v17_own_dynamic_litter_head10_manual「同影像同標註」的整份複製，
#     多疊一層 pre-roll BGS changemap 當第 4 通道。5 類：car/litter/motorcycle/person/garbage。
#
# 實驗定位：把「BGS 4ch 早期融合」疊到第 16 次「同一份 v17 手動配方」上，
#   單變因對照 = 第 16 次純 RGB（co_dino_chiayi_v17_own_dynamic_head10_manual_36e）。
#   成敗以動態子集 test_dynamic166 為主判定、static594 副（雙軌都報）。BGS 傍證見第 11/14 次。
#
# 相對第 16 次（純 RGB）只改「輸入變 4 通道 + 讓遮罩通道學得到」：
#   ① backbone in_channels 3→4；載入 conv1 已平均初始化的 4ch 預訓權重
#   ② frozen_stages 1→-1：解凍 stem(conv1)+layer1，否則第 4 通道投影永遠停在 init（第 11 次失敗主因）
#   ③ paramwise 新增 backbone.conv1 lr_mult=1.0：遮罩通道是新東西，需大 lr 從頭學
#   ④ LoadImageAndChangemap + Det4ChDataPreprocessor + PAD4：RGB+changemap 併成 4ch
#   ⑤ Conv1Ch4NormLogger 診斷探針（追蹤第 4 通道 L2 norm 是否在動）
#   ⑥ norm_eval 維持 True（同第 12/14 次；bs=1 下 BN 用凍結 running stats）
#
# ★ 與第 14 次的關鍵區別：backbone 主體 lr_mult = **0.01（=1e-6，同第 16 次）**，非第 14 次的 0.1。
#   → 只留「第 4 通道」當變因，backbone 主體與第 16 次逐字相同 → 比第 14 次更乾淨的單變因。
#   其餘（head 5 類、class_weight litter/garbage×3、ClassBalancedDataset oversample_thr=0.0、
#   filter_empty_gt=False、milestones[28,34]、36e、save_best=litter_precision、clip_grad、wd）一律同第 16 次。
#
# LR decay 路徑：
#   ep1-27    conv1 1e-4 / backbone(其餘) 1e-6 / head 1e-4
#   ep28-33   ×0.1
#   ep34-36   ×0.1
#
# BGS changemap 產生方式（見 experiments/18th_v18_own_dynamic_bgs_122vids/資料前處理/）：
#   build_alignment_v18.py（多斜率首幀對齊、支援小時制偏移）→ alignment_v18.json；
#   gen_changemaps_v18.py：pre-roll 凍結背景兩段式 KNN(500/500/shadow, rm5)，WARM=150 事件前幀建乾淨地面，
#   凍結後對 dataset jpg 取前景。1 部 0-preroll clip(20240713054438) 用尾巴暖機。
#   全 3522 張與影像檔名一一對應（train2494/valid373/test655）。逐 clip 記錄見 changemap_逐clip處理記錄_v18.md。
# ─────────────────────────────────────────────────────────────────────────────

custom_imports = dict(
    imports=[
        "projects.CO-DETR.codetr",
        "hooks.training_md_logger",
        "hooks.weighted_focal_loss",
        "hooks.load_changemap",
        "hooks.data_preprocessor_4ch",
        "hooks.conv1_ch4_norm_logger",
    ],
    allow_failed_imports=False,
)

_base_ = [
    "/opt/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_r50_lsj_8xb2_3x_coco.py"
]

data_root = "/workspace/datasets/coco/chiayi_v18_own_dynamic_bgs_122vids/"
NUM_CLASSES = 5
CLASSES = ("car", "litter", "motorcycle", "person", "garbage")

# FocalLoss / QualityFocalLoss（sigmoid multi-label，C=5）：長度 5
CLASS_WEIGHT = [1.0, 3.0, 1.0, 1.0, 3.0]
# CrossEntropyLoss（softmax，C=num_classes+1=6，含 background）：長度 6
CLASS_WEIGHT_CE = [1.0, 3.0, 1.0, 1.0, 3.0, 1.0]  # background weight=1.0

# ── head 5 類 + class_weight（同第 16 次）；backbone 4 通道解凍 + 4ch data_preprocessor ──
model = dict(
    backbone=dict(
        frozen_stages=-1,          # ② 全解凍：conv1(含第4通道)+layer1 可訓練
        norm_eval=True,            # ⑥ bs=1 下 BN 用凍結 running stats
        in_channels=4,             # ①
    ),
    data_preprocessor=dict(
        type="Det4ChDataPreprocessor",             # ④ 自訂 4 通道 mean/std
        mean=[123.675, 116.28, 103.53, 0.0],       # 第4通道(changemap) mean=0
        std=[58.395, 57.12, 57.375, 128.0],        # 255→~2.0，與 RGB 量級相當
        bgr_to_rgb=False,                          # LoadImageAndChangemap 已以 RGB 載入
        pad_mask=False,
        batch_augments=None,
    ),
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
)

# ── LSJ pipeline（4 通道：LoadImageAndChangemap + 4-tuple pad_val）────────────
# 純幾何增強、無 PhotoMetricDistortion → 不污染遮罩通道。RGB+遮罩併成 4ch 後一起經
# RandomResize/RandomCrop/RandomFlip/Pad，天然同步。
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

# ── Train dataloader（ClassBalancedDataset + persistent_workers；同第 16 次）──
# oversample_thr=0.0 → 對任何分布都保證不過採樣（v18 資料同 v17，dryrun 有 len==len assert 擋回歸）。
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
    dict(type="LoadImageAndChangemap"),
    dict(type="Resize", scale=image_size, keep_ratio=True),
    dict(type="Pad", size=image_size, pad_val=dict(img=PAD4)),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=False),
    dict(type="PackDetInputs",
         meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")),
]

# ── Val / Test dataloader（訓練期監控 valid litter_precision）──────────────────
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

# 同第 16 次：test 預設沿用 val（訓練期不評 test）。訓練後雙軌評估（dynamic166 主 / static594 副）
# 由 error_analysis / tools/test.py 以 --split-ann / --cfg-options 覆寫 ann_file 進行。
test_dataloader = val_dataloader
test_evaluator = val_evaluator

# ── 訓練排程（36 epoch，LR drop at 28 & 34；同第 16 次）──────────────────────
max_epochs = 36
train_cfg = dict(max_epochs=max_epochs)

param_scheduler = [
    dict(type="LinearLR", start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(type="MultiStepLR", begin=0, end=max_epochs, by_epoch=True,
         milestones=[28, 34], gamma=0.1),
]

# ── 差異化 LR（conv1 大 lr 學遮罩通道，其餘 backbone 0.01x 微調＝同第 16 次）──
# custom_keys「最長匹配」勝出：backbone.conv1.* → 1.0x（=1e-4，從頭學）；
#   其餘 backbone → 0.01x（=1e-6，同第 16 次，強力保護預訓練特徵）；head 三支 → 1.0x（=1e-4）。
optim_wrapper = dict(
    _delete_=True,
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            "backbone.conv1": dict(lr_mult=1.0),   # ③ 遮罩通道所在第一層，大 lr 從頭學
            "backbone": dict(lr_mult=0.01),        # ★ 0.01（=1e-6）＝同第 16 次，非第 14 次的 0.1
            "query_head": dict(lr_mult=1.0),
            "roi_head": dict(lr_mult=1.0),
            "bbox_head": dict(lr_mult=1.0),
        }
    ),
)

# ── Checkpoint：監控 litter precision（同第 16 次）──────────────────────────
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=5,
        save_best="coco/litter_precision",
        rule="greater",
    ),
)

# ── 詳細訓練記錄 hook + conv1 第4通道診斷探針 ──────────────────────────────
custom_hooks = [
    dict(
        type="TrainingMarkdownLogger",
        log_file="/workspace/hdd2_storage/18th_expt_v18_own_dynamic_bgs_4ch_36e/training_log.md",
        priority="VERY_LOW",
    ),
    dict(type="Conv1Ch4NormLogger"),   # ⑤ 追蹤 conv1 第4通道 L2 norm 是否在動
]

# ── 4ch 預訓權重（conv1 已 inflate 平均初始化），忽略 head ───────────────────
load_from = "/workspace/project/models/co_dino_5scale_lsj_r50_3x_coco_4ch_meaninit.pth"
load_from_excluding = ["query_head", "roi_head", "bbox_head"]

# ── 輸出路徑（寫 HDD，避免主碟）────────────────────────────────────────────
work_dir = "/workspace/hdd2_storage/18th_expt_v18_own_dynamic_bgs_4ch_36e"
