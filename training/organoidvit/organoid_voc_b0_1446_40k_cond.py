_base_ = [
    '../_base_/models/2DSegFomer.py',
    '../_base_/datasets/pascal_voc12.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k_adamw.py'
]

# Closer reproduction of the paper training protocol:
# 512x512 input crop, 40k steps, ImageNet-pretrained MiT-B0, and high-resolution
# resize before cropping/testing to preserve small organoid regions.
# Labels: 0 background, 1 live, 2 death, 3 bubble.
data_root = 'OrganoidSegBaseline/VOC2012'
crop_size = (512, 512)
img_scale = (1446, 1446)

norm_cfg = dict(type='SyncBN', requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    pretrained='pretrained/mit_b0.pth',
    backbone=dict(type='myt_b0', style='pytorch'),
    decode_head=dict(
        type='UHead',
        in_channels=[32, 64, 160, 256],
        in_index=[0, 1, 2, 3],
        feature_strides=[4, 8, 16, 32],
        channels=128,
        dropout_ratio=0.1,
        num_classes=4,
        norm_cfg=norm_cfg,
        align_corners=False,
        decoder_params=dict(embed_dim=256),
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomRotate', prob=0.5, degree=45, pad_val=0, seg_pad_val=255),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='PascalVOCDataset',
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/train_cond.txt',
        pipeline=train_pipeline),
    val=dict(
        type='PascalVOCDataset',
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/val_cond.txt',
        pipeline=test_pipeline),
    test=dict(
        type='PascalVOCDataset',
        data_root=data_root,
        img_dir='JPEGImages',
        ann_dir='SegmentationClass',
        split='ImageSets/Segmentation/test_cond.txt',
        pipeline=test_pipeline))

# Paper states Adam with lr=5e-6. The released codebase is AdamW-based; using
# Adam here to match the manuscript more closely.
optimizer = dict(_delete_=True, type='Adam', lr=0.000005, betas=(0.9, 0.999), weight_decay=0.0)
lr_config = dict(_delete_=True, policy='poly', power=1.0, min_lr=0.0, by_epoch=False)

runner = dict(type='IterBasedRunner', max_iters=40000)
evaluation = dict(interval=2000, metric=['mIoU', 'mDice'])
checkpoint_config = dict(by_epoch=False, interval=2000)
work_dir = 'OrganoidSegBaseline/runs/orgvit_b0_1446_40k_cond'
