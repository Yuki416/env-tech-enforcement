# Co-DINO 5scale + R50 ── 垃圾偵測 12 epoch Training Config
# ─────────────────────────────────────────────────────────────────────────────

_base_ = [
    "/opt/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_r50_8xb2_1x_coco.py"
]

# ── 資料集路徑（container 內）────────────────────────────────────────────────
data_root = "/workspace/datasets/coco/chiayi-violation-detection.v2-first-annotation--late-night--dawn-time.coco-mmdetection/"
NUM_CLASSES = 1

# ── 修改 head 的類別數 ────────────────────────────────────────────────────────
model = dict(
    query_head=dict(
        num_classes=NUM_CLASSES,
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
)

# ── 資料 pipeline ─────────────────────────────────────────────────────────────
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=("litter",)),
        ann_file="train/_annotations.coco.json",
        data_prefix=dict(img="train/"),
    ),
)

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=("litter",)),
        ann_file="valid/_annotations.coco.json",
        data_prefix=dict(img="valid/"),
    ),
)

val_evaluator = dict(
    ann_file=data_root + "valid/_annotations.coco.json"
)

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# ── 訓練排程（12 epoch）──────────────────────────────────────────────
max_epochs = 12
train_cfg = dict(max_epochs=max_epochs)

# 用 COCO 預訓練權重做 fine-tune，僅 head 不載入（類別數不同）
load_from = "/workspace/project/models/co_dino_5scale_r50_1x_coco-7481f903.pth"

# 讀取時忽略 shape 不符的 head 層
load_from_excluding = [
    "query_head", "roi_head", "bbox_head"
]

# ── 輸出路徑 ──────────────────────────────────────────────────────────────────
work_dir = "/workspace/project/results/chiayi_12e"
