import json
import os

PATH = r'D:\jht_code\mmdetection_git\mmdetection\work_dir\tood_r50_fpn_anchor_based_1x_tea\20230708_014738\vis_data'  # json文件所在路径
json_file = os.listdir(PATH)
source_file = None
new_file = None
for j in json_file:
    if j.endswith('.json'):
        source_file = PATH + '\\' + j
        new_file = PATH + '\\' + j.split('.')[0] + '_' + '.json'

    f = open(source_file, 'rb')
    # print(f.read())  # 读出整个文件
    # print(f.readlines())
    # print(len(f.readlines()))
    # print(type(f.read()))
    fw = open(new_file, 'wb')
    fw.write('['.encode('utf-8'))
    # fw.seek(1,0)
    for i in f.readlines():
        fw.write(i)
        fw.write(','.encode('utf-8'))
    # print(fw.tell())
    fw.seek(-1, 2)
    fw.write(']'.encode('utf-8'))
    fw.close()
    f.close()

# f = open(source_file,'rb')
# # print(f.read())  # 读出整个文件
# # print(f.readlines())
# # print(len(f.readlines()))
# # print(type(f.read()))
# fw = open(new_file,'wb')
# fw.write('['.encode('utf-8'))
# # fw.seek(1,0)
# for i in f.readlines():
#     fw.write(i)
#     fw.write(','.encode('utf-8'))
# # print(fw.tell())
# fw.seek(-1,2)
# fw.write(']'.encode('utf-8'))
# fw.close()
# f.close()