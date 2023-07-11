_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='TOOD',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='TOODHead',
        # num_classes=80,
        num_classes=1,
        in_channels=256,
        stacked_convs=6,
        feat_channels=256,
        anchor_type='anchor_free',
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        initial_loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    train_cfg=dict(
        initial_epoch=4,
        initial_assigner=dict(type='ATSSAssigner', topk=9),
        assigner=dict(type='TaskAlignedAssigner', topk=13),
        alpha=1,
        beta=6,
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001))


# 数据集相关配置
data_root = r'D:\jht_datasets\tea\img1_2'
metainfo = {
    'classes':('Sick',),
    'palette':[(220,20,60)]
}

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)

train_dataloader = dict(
    batch_size = 4,
    dataset=dict(data_root = data_root,
                 metainfo = metainfo,
                 ann_file = r'D:\jht_datasets\tea\img1_2\annotations\instance_train.json',
                 data_prefix = dict(img=r'D:\jht_datasets\tea\img1_2\train')
                 )
)
val_dataloader = dict(
    dataset = dict(
        data_root = data_root,
        metainfo = metainfo,
        ann_file = r'D:\jht_datasets\tea\img1_2\annotations\instance_val.json',
        data_prefix = dict(img=r'D:\jht_datasets\tea\img1_2\val')
    )
)

# test_dataloader = val_dataloader
test_dataloader = dict(dataset = dict(data_root = data_root,
                                      metainfo = metainfo,
                                      ann_file = r'D:\jht_datasets\tea\img1_2\annotations\instance_test.json',
                                      data_prefix = dict(img=r'D:\jht_datasets\tea\img1_2\test')))


# 修改评价指标的相关配置
val_evaluator = dict(ann_file = data_root + r'\annotations\instance_val.json')
# test_evaluator = val_evaluator
test_evaluator = dict(ann_file = data_root + r'\annotations\instance_test.json')