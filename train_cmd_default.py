#-*-coding:utf-8-*-

cmds = [
    'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 1.00E-03 --epochs 400 --LR 0.001 --optimize sgd --weights pretrained/26/best.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.002 --s_decay 1 --lr_decay 0 --wdir tmp_1 '
    'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 2.00E-03 --epochs 400 --LR 0.001 --optimize sgd --weights pretrained/26/best.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.002 --s_decay 1 --lr_decay 0 --wdir tmp_2 '
    'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 1.00E-03 --epochs 300 --LR 0.001 --optimize sgd --weights pretrained/26/best.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.005 --s_decay 1 --lr_decay 0 --wdir tmp_4 '
    'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 2.00E-03 --epochs 300 --LR 0.001 --optimize sgd --weights pretrained/26/best.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.005 --s_decay 1 --lr_decay 0 --wdir tmp_5 '
    'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 8.00E-04 --epochs 300 --LR 0.001 --optimize sgd --weights pretrained/26/best.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.005 --s_decay 1 --lr_decay 0 --wdir tmp_6 '

    'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 2.00E-03 --epochs 400 --LR 0.002 --optimize sgd --weights pretrained/26/best.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.002 --s_decay 1 --lr_decay 0 --wdir tmp_7 '
    'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 2.50E-03 --epochs 400 --LR 0.002 --optimize sgd --weights pretrained/26/best.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.002 --s_decay 1 --lr_decay 0 --wdir tmp_8 '
    'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 1.50E-03 --epochs 400 --LR 0.002 --optimize sgd --weights pretrained/26/best.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.002 --s_decay 1 --lr_decay 0 --wdir tmp_9 '
    'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 2.00E-03 --epochs 300 --LR 0.002 --optimize sgd --weights pretrained/26/best.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.005 --s_decay 1 --lr_decay 0 --wdir tmp_10 '
    'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 2.50E-03 --epochs 300 --LR 0.002 --optimize sgd --weights pretrained/26/best.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.005 --s_decay 1 --lr_decay 0 --wdir tmp_11 '
    'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 1.50E-03 --epochs 300 --LR 0.002 --optimize sgd --weights pretrained/26/best.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.005 --s_decay 1 --lr_decay 0 --wdir tmp_12 '

    'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 3.00E-03 --epochs 400 --LR 0.003 --optimize sgd --weights pretrained/26/best.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.002 --s_decay 1 --lr_decay 0 --wdir tmp_7 '
    'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 3.50E-03 --epochs 400 --LR 0.003 --optimize sgd --weights pretrained/26/best.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.002 --s_decay 1 --lr_decay 0 --wdir tmp_8 '
    'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 2.00E-03 --epochs 400 --LR 0.003 --optimize sgd --weights pretrained/26/best.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.002 --s_decay 1 --lr_decay 0 --wdir tmp_9 '
    'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 3.00E-03 --epochs 300 --LR 0.003 --optimize sgd --weights pretrained/26/best.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.005 --s_decay 1 --lr_decay 0 --wdir tmp_10 '
    'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 3.50E-03 --epochs 300 --LR 0.003 --optimize sgd --weights pretrained/26/best.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.005 --s_decay 1 --lr_decay 0 --wdir tmp_11 '
    'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 2.00E-03 --epochs 300 --LR 0.003 --optimize sgd --weights pretrained/26/best.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.005 --s_decay 1 --lr_decay 0 --wdir tmp_12 '

    'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 5.00E-04 --epochs 400 --LR 0.0005 --optimize sgd --weights pretrained/26/best.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.002 --s_decay 1 --lr_decay 0 --wdir tmp_13 '
    'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 8.00E-04 --epochs 400 --LR 0.0005 --optimize sgd --weights pretrained/26/best.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.002 --s_decay 1 --lr_decay 0 --wdir tmp_14 '
    'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 5.00E-04 --epochs 300 --LR 0.0005 --optimize sgd --weights pretrained/26/best.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.005 --s_decay 1 --lr_decay 0 --wdir tmp_15 '
    'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 8.00E-04 --epochs 300 --LR 0.0005 --optimize sgd --weights pretrained/26/best.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.005 --s_decay 1 --lr_decay 0 --wdir tmp_16 '

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
    if '--sr True' in cmd:
        cmd = cmd.replace('--sr True', '-sr')
    if '--sr False' in cmd:
        cmd = cmd.replace('--sr False', '')
    return cmd

import os
log = open("train_log.log", "a+")
for cmd in cmds:
    log.write(cmd)
    log.write("\n")
    cmd=check_name(cmd)
    os.system(cmd)
