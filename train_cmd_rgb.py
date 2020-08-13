#-*-coding:utf-8-*-

cmds = [
    #" CUDA_VISIBLE_DEVICES=0 python train.py --wdir black/spp/black_416_w_500 --cfg cfg/yolov3-spp-1cls.cfg --data data/swim_enhanced/enhanced.data --weights weights/yolov3-spp.weights --batch-size 32 --epochs 500",
    #" CUDA_VISIBLE_DEVICES=0 python train.py --wdir black/spp/black_416_500 --cfg cfg/yolov3-spp-1cls.cfg --data data/swim_enhanced/enhanced.data --batch-size 32 --epochs 500",
    #" CUDA_VISIBLE_DEVICES=0 python train.py --wdir rgb/spp_300_16_w_fr --cfg cfg/yolov3-spp-1cls.cfg --data data/rgb/rgb.data --batch-size 16 --weights weights/yolov3-spp.weights --epochs 300 --freeze",
    #" CUDA_VISIBLE_DEVICES=0 python train.py --wdir rgb/spp/200_16_dark_yes_416_multi --cfg cfg/yolov3-spp-1cls.cfg --multi-scale --img_size 416 --weights weights/darknet53.conv.74 --data data/rgb/rgb.data --batch-size 16  --freeze --epochs 300 ",
 #" CUDA_VISIBLE_DEVICES=0 python train.py --wdir rgb/spp/200_16_dark_yes_416_multi --cfg cfg/yolov3-spp-1cls.cfg --multi-scale --img_size 416 --accumulate 4 --weights weights/darknet53.conv.74 --data data/rgb/rgb.data --batch-size 16  --freeze --epochs 200 ",
 " CUDA_VISIBLE_DEVICES=1 python train.py --wdir rgb/spp/150_16_dark_yes_608_multi --cfg cfg/yolov3-spp-1cls.cfg --multi-scale --img_size 608 --accumulate 4 --weights weights/darknet53.conv.74 --data data/rgb/rgb.data --batch-size 8  --freeze --epochs 200 ",

]

import os
for cmd in cmds:
    os.system(cmd)
