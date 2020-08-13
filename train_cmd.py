#-*-coding:utf-8-*-

cmds = [
    # " CUDA_VISIBLE_DEVICES=2 python train.py --wdir gray/gray_416_w --cfg cfg/yolov3-1cls.cfg --data data/swim_gray/gray.data --weights weights/darknet53.conv.74 --batch-size 16 --epochs 300",
    # " CUDA_VISIBLE_DEVICES=2 python train.py --wdir gray/gray_416 --cfg cfg/yolov3-1cls.cfg --data data/swim_gray/gray.data --batch-size 16 --epochs 300",
    # " CUDA_VISIBLE_DEVICES=2 python train.py --wdir gray/gray_608_w --cfg cfg/yolov3-1cls-608.cfg --data data/swim_gray/gray.data --img_size 608 --weights weights/darknet53.conv.74 --batch-size 16 --epochs 300",
    # " CUDA_VISIBLE_DEVICES=2 python train.py --wdir gray/gray_608 --cfg cfg/yolov3-1cls-608.cfg --data data/swim_gray/gray.data --img_size 608 --batch-size 16 --epochs 300",
    # "python train.py --wdir gray_sE-3 --cfg cfg/yolov3-1cls.cfg --data data/swim_gray/gray.data --weights weights/gray_sE-3/last.pt --batch-size 32 --epochs 300 -sr --s 0.001 --prune 1",
    # "python train.py --wdir gray_s2E-3 --cfg cfg/yolov3-1cls.cfg --data data/swim_gray/gray.data --weights weights/gray_s2E-3/last.pt --batch-size 32 --epochs 300 -sr --s 0.002 --prune 1",
    # "python train.py --wdir gray_s5E-3 --cfg cfg/yolov3-1cls.cfg --data data/swim_gray/gray.data --weights weights/darknet53.conv.74 --batch-size 32 --epochs 300 -sr --s 0.005 --prune 1",
    # "python train.py --wdir gray_s3E-3 --cfg cfg/yolov3-1cls.cfg --data data/swim_gray/gray.data --weights weights/darknet53.conv.74 --batch-size 32 --epochs 300 -sr --s 0.003 --prune 1",
    # "python train.py --wdir gray_s2E-3_0.2*0.01 --cfg cfg/yolov3-1cls.cfg --data data/swim_gray/gray.data --weights weights/darknet53.conv.74 --batch-size 32 --epochs 300 -sr --s 0.002 --prune 1",
    # "python train.py --wdir gray_sE-3_0.35*0.01 --cfg cfg/yolov3-1cls.cfg --data data/swim_gray/gray.data --weights weights/gray_sE-3_0.35*0.01/last.pt --batch-size 32 --epochs 300 -sr --s 0.001 --prune 1",
    # "python train.py --wdir gray_sE-3_partS_0.35*0.01 --cfg cfg/yolov3-1cls.cfg --data data/swim_gray/gray.data --weights weights/gray_sE-3_0.35*0.01/last.pt --batch-size 32 --epochs 300 -sr --s 0.001 --prune 1",
    # "python train.py --wdir gray_sE-3_0.35*0.01 --cfg cfg/yolov3-1cls.cfg --data data/swim_gray/gray.data --weights weights/gray_sE-3_0.35*0.01/last.pt --batch-size 32 --epochs 300 -sr --s 0.001 --prune 1",
    # "python train.py --wdir ceiling_0527 --cfg cfg/yolov3-1cls.cfg --data data/ceiling/ceiling.data --weights weights/darknet53.conv.74 --batch-size 32 --epochs 100",
    # "python train.py --wdir ceiling_0527_300 --cfg cfg/yolov3-1cls.cfg --data data/ceiling/ceiling.data --weights weights/darknet53.conv.74 --batch-size 32 --epochs 300",
   # " CUDA_VISIBLE_DEVICES=3 python train.py --wdir gray/gray_416_500 --cfg cfg/yolov3-1cls.cfg --data data/swim_gray/gray.data --batch-size 32 --epochs 500",
    " CUDA_VISIBLE_DEVICES=3 python train.py --wdir gray/gray_608_w_500 --cfg cfg/yolov3-1cls-608.cfg --data data/swim_gray/gray.data --img_size 608 --weights weights/darknet53.conv.74 --batch-size 16 --epochs 500",
    " CUDA_VISIBLE_DEVICES=3 python train.py --wdir gray/gray_608_500 --cfg cfg/yolov3-1cls-608.cfg --data data/swim_gray/gray.data --img_size 608 --batch-size 16 --epochs 500",
]

import os
for cmd in cmds:
    os.system(cmd)
