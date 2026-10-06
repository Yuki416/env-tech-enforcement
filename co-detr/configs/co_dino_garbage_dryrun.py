# Co-DINO 5scale + R50 ── 垃圾偵測 Dry-Run Config
# ─────────────────────────────────────────────────────────────────────────────
# 繼承官方 COCO 設定，僅修改：
#   1. data_root   → 自定義資料集路徑
#   2. num_classes → 2（garbage / cigarette_butt）
#   3. max_epochs  → 1（Dry-Run 驗證 pipeline）
#   4. batch_size  → 2（避免 OOM；A6000 有 48 GB 可視需要調高）
# ─────────────────────────────────────────────────────────────────────────────

_base_ = [
    "/opt/mmdetection/projects/CO-DETR/configs/codino/co_dino_5scale_r50_8xb2_1x_coco.py"
]

# ── 資料集路徑（container 內）────────────────────────────────────────────────
data_root = "/workspace/datasets/custom/"    # Roboflow 下載後放這裡
NUM_CLASSES = 2

# ── 修改 head 的類別數 ────────────────────────────────────────────────────────
model = dict(
    query_head=dict(
        num_classes=NUM_CLASSES,
        transformer_head=dict(
            decoder=dict(num_classes=NUM_CLASSES),
        ),
    ),
    roi_head=[
        dict(
            type="CoStandardRoIHead",
            bbox_head=dict(type="CoConvFCBBoxHead", num_classes=NUM_CLASSES),
        )
    ],
    bbox_head=[
        dict(type="CoATSSHead", num_classes=NUM_CLASSES),
        dict(type="CoATSSHead", num_classes=NUM_CLASSES),
        dict(type="CoATSSHead", num_classes=NUM_CLASSES),
    ],
)

# ── 資料 pipeline ─────────────────────────────────────────────────────────────
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=("garbage", "cigarette_butt")),
        ann_file="train/_annotations.coco.json",
        data_prefix=dict(img="train/"),
    ),
)

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=dict(classes=("garbage", "cigarette_butt")),
        ann_file="valid/_annotations.coco.json",
        data_prefix=dict(img="valid/"),
    ),
)

val_evaluator = dict(
    ann_file=data_root + "valid/_annotations.coco.json"
)

test_dataloader = val_dataloader
test_evaluator = val_evaluator

# ── 訓練排程（Dry-Run: 1 epoch）──────────────────────────────────────────────
max_epochs = 1
train_cfg = dict(max_epochs=max_epochs)

# 用 COCO 預訓練權重做 fine-tune，僅 head 不載入（類別數不同）
load_from = "/workspace/project/models/co_dino_5scale_r50_1x_coco-7481f903.pth"

# 讀取時忽略 shape 不符的 head 層
load_from_excluding = [
    "query_head", "roi_head", "bbox_head"
]

# ── 輸出路徑 ──────────────────────────────────────────────────────────────────
work_dir = "/workspace/project/results/dryrun"
