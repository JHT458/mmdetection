_base_ = './tood_r50_fpn_1x_tea_1.py'
model = dict(bbox_head=dict(anchor_type='anchor_based'))
