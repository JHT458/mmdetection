_base_ = ['../rpn/rpn_r50_fpn_1x_coco.py']

model = dict(roi_head = dict(bbox_head = dict(num_classes=1)))

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
# 使用预训练的模型权重来做初始化，提高模型性能
# load_from = ''