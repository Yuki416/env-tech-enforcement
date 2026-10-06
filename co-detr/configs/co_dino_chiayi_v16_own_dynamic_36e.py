# Co-DINO 5scale + R50 LSJ 3x ── 第十五次實驗：v16 動態版（121 部影片的動態子集）
# 資料集：chiayi_v16_own_dynamic_litter
#   由 v16(121部) 過濾而來，配方與第十三次 v14 **逐字相同**：
#   litter(2)+garbage(5) 各自用「相鄰幀 GT 位移 >=20px/幀」判定動態種子，再把同 clip 連續影格納入：
#   頭段第一個 seed 往前 30 幀、尾段最後一個 seed 往後 5 幀（HEAD_WINDOW=30 / TAIL_WINDOW=5）。
#   train 3009 幀(99 clip) / valid 488 幀(14 clip)（含 ~46% negative 填充幀）；
#   test 用 v16 完整版 655 幀(8 clip)，不過濾。
#
# 目的：檢驗「訓練事件多樣性不足」是否為第十三次表現受限的主因。
#   第十三次 train 僅 44 個丟垃圾事件，litter AP train 0.827 → test 0.226（典型小資料過擬合）。
#   本次把事件數拉到 99，**其餘一切不動**。
#
# ★ 對照 baseline = 第十三次 v14。**第十三次的訓練集是本次的嚴格子集**（已逐檔名驗證）：
#   train v11 部分 1237 幀與第十三次完全相同 + 新增 1772 幀；valid 488 幀完全相同。
#   → 單一變因 = 訓練資料規模（1237→3009 幀 / 44→99 clips）。
#
# 改動（相對第十三次 v14）：
#   1. data_root → chiayi_v16_own_dynamic_litter
#   2. ClassBalancedDataset oversample_thr 0.1 → 0.0（**保持行為一致，非改變策略**，見下方說明）
#   3. work_dir / training_log 路徑 → 15th_expt_...
#   其餘（LR、class_weight、milestones、36e、filter_empty_gt=False、save_best）完全沿用。
#
# ★★ 為什麼 oversample_thr 要從 0.1 改成 0.0（重要，報告需說明）：
#   MMEngine ClassBalancedDataset：r(c) = max(1, sqrt(thr/f(c)))，再對每張圖取 **math.ceil(r)**。
#     第十三次 v14：garbage f=0.165 → r=1.0 → ceil=1 → **不複製（空轉）**
#     第十五次 v16：garbage f=**0.0994** → r=**1.0032** → ceil=**2** → **含 garbage 的 299 張整份複製**
#   garbage 的影像佔比剛好從 0.165 掉到 0.0994，差 0.0006 跨過門檻；而 ceil 把「多 0.3%」
#   進位成「多 100%」。若照抄 0.1，本次就會變成「資料量 2.43× **且** garbage 過採樣 2×」＝ 多一個 confound。
#   設 thr=0.0 → sqrt(0/f)=0 → r=max(1.0, 0)=1.0 → ceil=1，**對任何資料分布都保證不過採樣**，
#   忠實重現第十三次的**實際行為**（設 0.05 只是「目前剛好」低於 0.0994，不穩固）。
#   使用者決定：本次不做 oversample。
#
# 注意：train/valid 為動態子集、test 為完整版（含靜止落地垃圾）→ 語意落差是刻意保留的，
#       主判定請用 annotations/test_static594.json（與第十三次同一把尺）。
#       v16 完整 655 test 含已知同片洩漏（見實驗夾 README），引用時須標註虛高。
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

data_root = "/workspace/datasets/coco/chiayi_v16_own_dynamic_litter/"
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
# oversample_thr=0.0 → 保證不過採樣（理由見檔頭★★）。保留 wrapper 以與第十三次維持最小結構差異。
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
# ⚠️ valid 與第十三次完全相同（14 部、0 部新增午後 clip）→ 選 ckpt 的機制與第十三次一致，
#    但對新增的午後 domain 無代表性。此限制已知並記錄於實驗夾 README。
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
        log_file="/workspace/hdd2_storage/15th_expt_v16_own_dynamic_121vids_36e/training_log.md",
        priority="VERY_LOW",
    )
]

# ── Co-DINO 3x COCO 預訓練權重，忽略 head 層（隨機初始化）────────────────────
load_from = "/workspace/project/models/co_dino_5scale_lsj_r50_3x_coco-fe5a6829.pth"
load_from_excluding = ["query_head", "roi_head", "bbox_head"]

# ── 輸出路徑 ──────────────────────────────────────────────────────────────────
# 結果直接寫 HDD（容器內 /workspace/hdd2_storage = host /path/to/data）
# 避免占用近滿的主碟；results/ 另有 host 端 symlink 供瀏覽。
work_dir = "/workspace/hdd2_storage/15th_expt_v16_own_dynamic_121vids_36e"
