#-*-coding:utf-8-*-

cmds = [
    #" CUDA_VISIBLE_DEVICES=0 python train.py --wdir black/spp/black_416_w_500 --cfg cfg/yolov3-spp-1cls.cfg --data data/swim_enhanced/enhanced.data --weights weights/yolov3-spp.weights --batch-size 32 --epochs 500",
    #" CUDA_VISIBLE_DEVICES=0 python train.py --wdir black/spp/black_416_500 --cfg cfg/yolov3-spp-1cls.cfg --data data/swim_enhanced/enhanced.data --batch-size 32 --epochs 500",
#'CUDA_VISIBLE_DEVICES=1 python train.py --wdir gray/spp/200_16_dark_yes_608_multi_150 --cfg cfg/yolov3-spp-1cls.cfg --multi-scale --freeze --data data/swim_gray/gray.data --batch-size 16 --weights weights/darknet53.conv.74 --epochs 150',
#'CUDA_VISIBLE_DEVICES=1 python train.py --wdir gray/spp/200_16_dark_yes_608_multi_250 --cfg cfg/yolov3-spp-1cls.cfg --multi-scale --freeze --data data/swim_gray/gray.data --batch-size 16 --weights weights/darknet53.conv.74 --epochs 250',
#'CUDA_VISIBLE_DEVICES=1 python train.py --wdir gray/spp/multi/200_416_yes_yes_mish --cfg cfg/yolov3-spp-1cls-mish.cfg --multi-scale --freeze --data data/swim_gray/gray.data --batch-size 4 --weights weights/darknet53.conv.74 --epochs 200',
#'CUDA_VISIBLE_DEVICES=0 python train.py --wdir gray/spp/multi/200_416_yes_yes_leaky --cfg cfg/yolov3-spp-1cls.cfg --multi-scale --freeze --data data/swim_gray/gray.data --batch-size 8 --weights weights/darknet53.conv.74 --epochs 200'
#'CUDA_VISIBLE_DEVICES=1 python train.py --wdir gray/spp//multi/200_16_dark_yes_416_multi_250 --cfg cfg/yolov3-spp-1cls.cfg --multi-scale --freeze --data data/swim_gray/gray.data --batch-size 16 --weights weights/darknet53.conv.74 --epochs 250',
#'CUDA_VISIBLE_DEVICES=1 python train.py --wdir gray/spp/multi/200_16_dark_yes_416_multi_200 --cfg cfg/yolov3-spp-1cls.cfg --multi-scale --arc default --freeze --data data/swim_gray/gray.data --batch-size 16 --weights weights/darknet53.conv.74 --epochs 200',
#'CUDA_VISIBLE_DEVICES=2 python train.py --wdir gray/spp/multi/200_16_dark_yes_416_multi_200_giou_cls --cfg cfg/yolov3-spp-1cls.cfg --multi-scale  --freeze --data data/swim_gray/gray.data --batch-size 16 --weights weights/darknet53.conv.74 --epochs 200',
#'CUDA_VISIBLE_DEVICES=1 python train.py --wdir gray/spp/activation/leaky --cfg cfg/yolov3-spp-1cls.cfg --multi-scale --data data/swim_gray/gray.data --batch-size 8 --epochs 150',
#'CUDA_VISIBLE_DEVICES=1 python train.py --wdir gray/spp/activation/swish --cfg cfg/yolov3-spp-1cls-swish.cfg --multi-scale --data data/swim_gray/gray.data --batch-size 8 --epochs 150',
#'CUDA_VISIBLE_DEVICES=1 python train.py --wdir gray/spp/activation/mish --cfg cfg/yolov3-spp-1cls-mish.cfg --multi-scale --data data/swim_gray/gray.data --batch-size 4 --epochs 150',
#'CUDA_VISIBLE_DEVICES=1 python train.py --wdir gray/spp/200_16_dark_yes_608_multi --cfg cfg/yolov3-spp-1cls.cfg --multi-scale --freeze --data data/swim_gray/gray.data --batch-size 16 --weights weights/darknet53.conv.74 --epochs 200',

#'CUDA_VISIBLE_DEVICES=0 python train.py --wdir gray/spp/multi/200_16_dark_no_416_multi_150 --cfg cfg/yolov3-spp-1cls.cfg --multi-scale --data data/swim_gray/gray.data --batch-size 16 --weights weights/darknet53.conv.74 --epochs 150',

'CUDA_VISIBLE_DEVICES=0 python train.py --wdir gray/spp/135_8_no_no_608_multi_prune0.9 --cfg cfg/prune_0.9_keep_0.1_15_shortcut.cfg --multi-scale --img_size 608 --data data/swim_gray/gray.data --batch-size 8 --epochs 135',




]

import os
for cmd in cmds:
    os.system(cmd)
