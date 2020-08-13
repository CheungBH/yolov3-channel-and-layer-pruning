#-*-coding:utf-8-*-

cmds = [
#'CUDA_VISIBLE_DEVICES=1 python train.py --wdir gray/spp//multi/200_16_dark_yes_416_multi_250 --cfg cfg/yolov3-spp-1cls.cfg --multi-scale --freeze --data data/swim_gray/gray.data --batch-size 16 --weights weights/darknet53.conv.74 --epochs 250',
#'CUDA_VISIBLE_DEVICES=1 python train.py --wdir gray/spp/multi/200_16_dark_yes_416_multi_200 --cfg cfg/yolov3-spp-1cls.cfg --multi-scale --arc default --freeze --data data/swim_gray/gray.data --batch-size 16 --weights weights/darknet53.conv.74 --epochs 200',
#'CUDA_VISIBLE_DEVICES=0 python train.py --wdir gray/spp/multi/200_16_dark_yes_416_multi_200_cls10 --cfg cfg/yolov3-spp-1cls.cfg --multi-scale  --freeze --data data/swim_gray/gray.data --batch-size 16 --weights weights/darknet53.conv.74 --epochs 200',
#'CUDA_VISIBLE_DEVICES=1 python train.py --wdir gray/spp/activation/leaky --cfg cfg/yolov3-spp-1cls.cfg --multi-scale --data data/swim_gray/gray.data --batch-size 8 --epochs 150',
#'CUDA_VISIBLE_DEVICES=1 python train.py --wdir gray/spp/activation/swish --cfg cfg/yolov3-spp-1cls-swish.cfg --multi-scale --data data/swim_gray/gray.data --batch-size 8 --epochs 150',
#'CUDA_VISIBLE_DEVICES=1 python train.py --wdir gray/spp/activation/mish --cfg cfg/yolov3-spp-1cls-mish.cfg --multi-scale --data data/swim_gray/gray.data --batch-size 8 --epochs 150',
#'CUDA_VISIBLE_DEVICES=0 python train.py --wdir black/spp/200_16_dark_yes_416_multi_150 --cfg cfg/yolov3-spp-1cls.cfg --multi-scale --freeze --data data/swim_black/black.data --batch-size 16 --weights weights/darknet53.conv.74  --epochs 150',
#'CUDA_VISIBLE_DEVICES=0 python train.py --wdir black/spp/200_16_dark_yes_416_multi_150_arc --cfg cfg/yolov3-spp-1cls.cfg --multi-scale --arc default --freeze --data data/swim_black/black.data --batch-size 16 --weights weights/darknet53.conv.74  --epochs 150',
#'CUDA_VISIBLE_DEVICES=2 python train.py --wdir black/spp/200_16_dark_yes_608_multi_150 --cfg cfg/yolov3-spp-1cls.cfg --multi-scale --img_size 608 --freeze --data data/swim_black/black.data --batch-size 8 --weights weights/darknet53.conv.74  --epochs 150',
#'CUDA_VISIBLE_DEVICES=2 python train.py --wdir black/spp/200_16_dark_yes_608_multi_150_arc --cfg cfg/yolov3-spp-1cls.cfg --multi-scale --arc default --img_size 608 --freeze --data data/swim_black/black.data --batch-size 8 --weights weights/darknet53.conv.74  --epochs 150',
'CUDA_VISIBLE_DEVICES=2 python train.py --wdir black/spp/200_16_dark_yes_416_multi_300 --cfg cfg/yolov3-spp-1cls.cfg --multi-scale --img_size 416 --freeze --data data/swim_black/black.data --batch-size 16 --weights weights/darknet53.conv.74  --epochs 300',
]

import os
for cmd in cmds:
    os.system(cmd)
