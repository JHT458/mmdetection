import threading
from threading import Thread
import time
from faster_rcnn_r50_fpn_1x_tea_2 import train as faster_rcnn_2_train
from faster_rcnn_r50_fpn_1x_tea_3 import train as faster_rcnn_3_train
from rpn_r50_fpn_1x_tea_1 import train as rpn_1_train
from rpn_r50_fpn_1x_tea_2 import train as rpn_2_train
from rpn_r50_fpn_1x_tea_3 import train as rpn_3_train
from retinanet_r18_fpn_1x_tea_1 import train as retinanet_1_train
from retinanet_r18_fpn_1x_tea_2 import train as retinanet_2_train
from retinanet_r18_fpn_1x_tea_3 import train as retinanet_3_train

from mask_rcnn_r50_fpn_1x_tea_1 import train as mask_rcnn_1_train
from tood_r50_fpn_1x_tea_1 import train as tood_1_train
from tood_r50_fpn_anchor_based_1x_tea import train as tood_anchor_based_1_train

from yolov3_d53_8xb8_320_273e_tea_1 import train as yolov3_d53_8xb8_320_273e_tea_1_tarin
from yolov3_d53_8xb8_amp_ms_608_273e_tea_1 import train as yolov3_d53_8xb8_amp_ms_608_273e_tea_1_train
from yolov3_d53_8xb8_ms_416_273e_tea_1 import train as yolov3_d53_8xb8_ms_416_273e_tea_1_train
from yolov3_d53_8xb8_ms_608_273e_tea_1 import train as yolov3_d53_8xb8_ms_608_273e_tea_1_train
from yolov3_mobilenetv2_8xb24_ms_416_300e_tea_1 import train as yolov3_mobilenetv2_8xb2_ms_416_300e_tea_1_train
from yolov3_mobilenetv2_8xb24_320_300e_tea_1 import train as yolov3_mobilenetv2_8xb24_320_300e_tea_1_train


# net_list = [faster_rcnn_2_train, faster_rcnn_3_train, rpn_1_train, rpn_2_train, rpn_3_train, retinanet_1_train, retinanet_2_train, retinanet_3_train]
# net_list = [mask_rcnn_1_train, tood_1_train, tood_anchor_based_1_train]
net_list = [yolov3_d53_8xb8_320_273e_tea_1_tarin, yolov3_d53_8xb8_amp_ms_608_273e_tea_1_train, yolov3_d53_8xb8_ms_416_273e_tea_1_train, yolov3_d53_8xb8_ms_608_273e_tea_1_train
            , yolov3_mobilenetv2_8xb2_ms_416_300e_tea_1_train, yolov3_mobilenetv2_8xb24_320_300e_tea_1_train]
lock = threading.Lock()
# for i in net_list:
#     print('正在训练网络：',i)
#     i.main()
#     print('==================================================================================')
#     print('==================================================================================')
def train(net):
    global lock
    lock.acquire()
    time.sleep(3)
    net.main()
    time.sleep(3)
    lock.release()


def main():
    for i in net_list:
        print('==================================================================================')
        print('正在训练网络：',i)
        print('==================================================================================')

        t = Thread(target=train(i))   # 创建线程实例
        t.start()  # 线程开始
        t.join()  # 等待线程结束，否则一直挂起，不需要等待时可以不用该join

        print('==================================================================================')
        print(i,' 已经训练结束')
        print('==================================================================================')

if __name__ == '__main__':
    main()