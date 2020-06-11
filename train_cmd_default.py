#-*-coding:utf-8-*-

cmds = [
    # "python train.py --wdir black_origin --cfg cfg/yolov3-1cls.cfg --data data/swim_enhanced/enhanced.data --weights weights/black_origin/last.pt --batch-size 16 --epochs 300",
    # "python train.py --wdir black_sE-3 --cfg cfg/yolov3-1cls.cfg --data data/swim_enhanced/enhanced.data --weights weights/black_sE-3/last.pt --batch-size 16 --epochs 300 -sr --s 0.001 --prune 1",
    # "python train.py --wdir black_s2E-3 --cfg cfg/yolov3-1cls.cfg --data data/swim_enhanced/enhanced.data --weights weights/black_s2E-3/last.pt --batch-size 16 --epochs 300 -sr --s 0.002 --prune 1",
    # "python train.py --wdir black_s5E-3 --cfg cfg/yolov3-1cls.cfg --data data/swim_enhanced/enhanced.data --weights weights/darknet53.conv.74 --batch-size 16 --epochs 300 -sr --s 0.005 --prune 1",
    # "python train.py --wdir black_s3E-3 --cfg cfg/yolov3-1cls.cfg --data data/swim_enhanced/enhanced.data --weights weights/darknet53.conv.74 --batch-size 16 --epochs 300 -sr --s 0.003 --prune 1",
    # "python train.py --wdir black_s2E-3_45*0.01 --cfg cfg/yolov3-1cls.cfg --data data/swim_enhanced/enhanced.data --weights weights/black_s2E-3_45*0.01/last.pt --batch-size 16 --epochs 300 -sr --s 0.002 --prune 1",
    # "python train.py --wdir black_s1E-3_60*0.01 --cfg cfg/yolov3-1cls.cfg --data data/swim_enhanced/enhanced.data --weights weights/darknet53.conv.74 --batch-size 16 --epochs 300 -sr --s 0.001 --prune 1",
    #"python train.py --wdir celiling_origin --cfg cfg/yolov3-1cls.cfg --data data/ceiling_cam/ceiling.data --weights weights/darknet53.conv.74 --batch-size 16 --epochs 100",
    # "python train.py --wdir ceiling0507_lrE-4 --cfg cfg/yolov3-1cls.cfg --data data/ceiling.data --weights weights/darknet53.conv.74 --batch-size 16 --epochs 100",
    "python train.py --wdir black_nopre --cfg cfg/yolov3-1cls.cfg --data data/swim_enhanced/enhanced.data --batch-size 16 --epochs 300",
    "python train.py --wdir gray_nopre --cfg cfg/yolov3-1cls.cfg --data data/swim_gray/gray.data --batch-size 16 --epochs 300",
]

import os
for cmd in cmds:
    os.system(cmd)
