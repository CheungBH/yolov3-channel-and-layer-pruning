#-*-coding:utf-8-*-

cmds = [
'CUDA_VISIBLE_DEVICES=3 python train.py --type original --activation swish --batch-size 4 --freeze False --epochs 150 --LR 0.00025 --optimize adam --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale True --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 251',
'CUDA_VISIBLE_DEVICES=3 python train.py --type tiny --activation swish --batch-size 4 --freeze False --epochs 150 --LR 0.00025 --optimize adam --weights weights/yolov3-tiny.conv.15 --save_interval 10 --multi-scale True --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 252',
'CUDA_VISIBLE_DEVICES=3 python train.py --type spp --activation leaky --batch-size 8 --freeze True --epochs 150 --LR 0.00025 --optimize sgd --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 253',
'CUDA_VISIBLE_DEVICES=3 python train.py --type original --activation leaky --batch-size 8 --freeze True --epochs 150 --LR 0.00025 --optimize sgd --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 254',
'CUDA_VISIBLE_DEVICES=3 python train.py --type tiny --activation leaky --batch-size 8 --freeze True --epochs 150 --LR 0.00025 --optimize sgd --weights weights/yolov3-tiny.conv.15 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 255',
'CUDA_VISIBLE_DEVICES=3 python train.py --type spp --activation swish --batch-size 4 --freeze True --epochs 150 --LR 0.00025 --optimize sgd --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 256',
'CUDA_VISIBLE_DEVICES=3 python train.py --type original --activation swish --batch-size 4 --freeze True --epochs 150 --LR 0.00025 --optimize sgd --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 257',
'CUDA_VISIBLE_DEVICES=3 python train.py --type tiny --activation swish --batch-size 4 --freeze True --epochs 150 --LR 0.00025 --optimize sgd --weights weights/yolov3-tiny.conv.15 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 258',
'CUDA_VISIBLE_DEVICES=3 python train.py --type spp --activation swish --batch-size 4 --freeze True --epochs 150 --LR 0.00025 --optimize sgd --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 259',
'CUDA_VISIBLE_DEVICES=3 python train.py --type original --activation swish --batch-size 4 --freeze True --epochs 150 --LR 0.00025 --optimize sgd --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 260',
'CUDA_VISIBLE_DEVICES=3 python train.py --type tiny --activation swish --batch-size 4 --freeze True --epochs 150 --LR 0.00025 --optimize sgd --weights weights/yolov3-tiny.conv.15 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 261',
'CUDA_VISIBLE_DEVICES=3 python train.py --type spp --activation leaky --batch-size 8 --freeze False --epochs 150 --LR 0.00025 --optimize sgd --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 262',
'CUDA_VISIBLE_DEVICES=3 python train.py --type original --activation leaky --batch-size 8 --freeze False --epochs 150 --LR 0.00025 --optimize sgd --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 263',
'CUDA_VISIBLE_DEVICES=3 python train.py --type tiny --activation leaky --batch-size 8 --freeze False --epochs 150 --LR 0.00025 --optimize sgd --weights weights/yolov3-tiny.conv.15 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 264',
'CUDA_VISIBLE_DEVICES=3 python train.py --type spp --activation swish --batch-size 4 --freeze False --epochs 150 --LR 0.00025 --optimize sgd --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 265',
'CUDA_VISIBLE_DEVICES=3 python train.py --type original --activation swish --batch-size 4 --freeze False --epochs 150 --LR 0.00025 --optimize sgd --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 266',
'CUDA_VISIBLE_DEVICES=3 python train.py --type tiny --activation swish --batch-size 4 --freeze False --epochs 150 --LR 0.00025 --optimize sgd --weights weights/yolov3-tiny.conv.15 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 267',
'CUDA_VISIBLE_DEVICES=3 python train.py --type spp --activation swish --batch-size 4 --freeze False --epochs 150 --LR 0.00025 --optimize sgd --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 268',
'CUDA_VISIBLE_DEVICES=3 python train.py --type original --activation swish --batch-size 4 --freeze False --epochs 150 --LR 0.00025 --optimize sgd --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 269',
'CUDA_VISIBLE_DEVICES=3 python train.py --type tiny --activation swish --batch-size 4 --freeze False --epochs 150 --LR 0.00025 --optimize sgd --weights weights/yolov3-tiny.conv.15 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 270',
'CUDA_VISIBLE_DEVICES=3 python train.py --type spp --activation leaky --batch-size 8 --freeze True --epochs 150 --LR 0.00025 --optimize adam --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 271',
'CUDA_VISIBLE_DEVICES=3 python train.py --type original --activation leaky --batch-size 8 --freeze True --epochs 150 --LR 0.00025 --optimize adam --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 272',
'CUDA_VISIBLE_DEVICES=3 python train.py --type tiny --activation leaky --batch-size 8 --freeze True --epochs 150 --LR 0.00025 --optimize adam --weights weights/yolov3-tiny.conv.15 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 273',
'CUDA_VISIBLE_DEVICES=3 python train.py --type spp --activation swish --batch-size 4 --freeze True --epochs 150 --LR 0.00025 --optimize adam --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 274',
'CUDA_VISIBLE_DEVICES=3 python train.py --type original --activation swish --batch-size 4 --freeze True --epochs 150 --LR 0.00025 --optimize adam --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 275',
'CUDA_VISIBLE_DEVICES=3 python train.py --type tiny --activation swish --batch-size 4 --freeze True --epochs 150 --LR 0.00025 --optimize adam --weights weights/yolov3-tiny.conv.15 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 276',
'CUDA_VISIBLE_DEVICES=3 python train.py --type spp --activation swish --batch-size 4 --freeze True --epochs 150 --LR 0.00025 --optimize adam --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 277',
'CUDA_VISIBLE_DEVICES=3 python train.py --type original --activation swish --batch-size 4 --freeze True --epochs 150 --LR 0.00025 --optimize adam --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 278',
'CUDA_VISIBLE_DEVICES=3 python train.py --type tiny --activation swish --batch-size 4 --freeze True --epochs 150 --LR 0.00025 --optimize adam --weights weights/yolov3-tiny.conv.15 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 279',
'CUDA_VISIBLE_DEVICES=3 python train.py --type spp --activation leaky --batch-size 8 --freeze False --epochs 150 --LR 0.00025 --optimize adam --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 280',
'CUDA_VISIBLE_DEVICES=3 python train.py --type original --activation leaky --batch-size 8 --freeze False --epochs 150 --LR 0.00025 --optimize adam --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 281',
'CUDA_VISIBLE_DEVICES=3 python train.py --type tiny --activation leaky --batch-size 8 --freeze False --epochs 150 --LR 0.00025 --optimize adam --weights weights/yolov3-tiny.conv.15 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 282',
'CUDA_VISIBLE_DEVICES=3 python train.py --type spp --activation swish --batch-size 4 --freeze False --epochs 150 --LR 0.00025 --optimize adam --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 283',
'CUDA_VISIBLE_DEVICES=3 python train.py --type original --activation swish --batch-size 4 --freeze False --epochs 150 --LR 0.00025 --optimize adam --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 284',
'CUDA_VISIBLE_DEVICES=3 python train.py --type tiny --activation swish --batch-size 4 --freeze False --epochs 150 --LR 0.00025 --optimize adam --weights weights/yolov3-tiny.conv.15 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 285',
'CUDA_VISIBLE_DEVICES=3 python train.py --type spp --activation swish --batch-size 4 --freeze False --epochs 150 --LR 0.00025 --optimize adam --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 286',
'CUDA_VISIBLE_DEVICES=3 python train.py --type original --activation swish --batch-size 4 --freeze False --epochs 150 --LR 0.00025 --optimize adam --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 287',
'CUDA_VISIBLE_DEVICES=3 python train.py --type tiny --activation swish --batch-size 4 --freeze False --epochs 150 --LR 0.00025 --optimize adam --weights weights/yolov3-tiny.conv.15 --save_interval 10 --multi-scale False --img_size 608 --rect True --data data/ceiling_train.data --expFolder gray	--expID 288',
]

def check_name(cmd):
    if '--multi-scale True' in cmd:
        cmd=cmd.replace('--multi-scale True','--multi-scale')
    if '--multi-scale False' in cmd:
        cmd = cmd.replace('--multi-scale False', '')
    if '--rect True' in cmd:
        cmd=cmd.replace('--rect True','--rect')
    if '--rect False' in cmd:
        cmd=cmd.replace('--rect False','')
    if '--freeze True' in cmd:
        cmd=cmd.replace('--freeze True','--freeze')
    if '--freeze False' in cmd:
        cmd=cmd.replace('--freeze False','')
    return cmd

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    cmd=check_name(cmd)
    os.system(cmd)
