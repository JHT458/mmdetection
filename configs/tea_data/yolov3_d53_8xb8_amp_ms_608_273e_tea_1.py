_base_ = './yolov3_d53_8xb8_ms_608_273e_tea_1.py'
# fp16 settings
optim_wrapper = dict(type='AmpOptimWrapper', loss_scale='dynamic')
