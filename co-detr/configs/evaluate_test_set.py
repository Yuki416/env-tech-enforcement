_base_ = './co_dino_litter_12e.py'
data_root = "/workspace/datasets/coco/chiayi-violation-detection.v2-first-annotation--late-night--dawn-time.coco-mmdetection/"

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=dict(classes=("litter",)),
        ann_file="test/_annotations.coco.json",
        data_prefix=dict(img="test/"),
        test_mode=True,
        pipeline=_base_.val_dataloader['dataset']['pipeline'] if 'pipeline' in _base_.val_dataloader['dataset'] else _base_.test_dataloader['dataset'].pipeline
    )
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + "test/_annotations.coco.json",
    metric='bbox',
    format_only=False
)
