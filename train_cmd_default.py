#-*-coding:utf-8-*-

cmds = [


'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 1.00E-03 --epochs 400 --LR 0.001 --optimize sgd --weights weights/tmp_1/last.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.002 --s_decay 0.1 --lr_decay 1 --wdir 0.1_1 ',
'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 2.00E-03 --epochs 400 --LR 0.001 --optimize sgd --weights weights/tmp_2/last.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.002 --s_decay 0.1 --lr_decay 1 --wdir 0.1_2 ',
'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 8.00E-04 --epochs 400 --LR 0.001 --optimize sgd --weights weights/tmp_3/last.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.002 --s_decay 0.1 --lr_decay 1 --wdir 0.1_3 ',
'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 1.00E-03 --epochs 300 --LR 0.001 --optimize sgd --weights weights/tmp_4/last.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.005 --s_decay 0.1 --lr_decay 1 --wdir 0.1_4 ',
'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 2.00E-03 --epochs 300 --LR 0.001 --optimize sgd --weights weights/tmp_5/last.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.005 --s_decay 0.1 --lr_decay 1 --wdir 0.1_5 ',
'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 8.00E-04 --epochs 300 --LR 0.001 --optimize sgd --weights weights/tmp_6/last.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.005 --s_decay 0.1 --lr_decay 1 --wdir 0.1_6 ',

'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 1.00E-03 --epochs 400 --LR 0.001 --optimize sgd --weights weights/tmp_1/last.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.002 --s_decay 0.01 --lr_decay 1 --wdir 0.01_1 ',
'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 2.00E-03 --epochs 400 --LR 0.001 --optimize sgd --weights weights/tmp_2/last.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.002 --s_decay 0.01 --lr_decay 1 --wdir 0.01_2 ',
'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 8.00E-04 --epochs 400 --LR 0.001 --optimize sgd --weights weights/tmp_3/last.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.002 --s_decay 0.01 --lr_decay 1 --wdir 0.01_3 ',
'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 1.00E-03 --epochs 300 --LR 0.001 --optimize sgd --weights weights/tmp_4/last.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.005 --s_decay 0.01 --lr_decay 1 --wdir 0.01_4 ',
'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 2.00E-03 --epochs 300 --LR 0.001 --optimize sgd --weights weights/tmp_5/last.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.005 --s_decay 0.01 --lr_decay 1 --wdir 0.01_5 ',
'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 8.00E-04 --epochs 300 --LR 0.001 --optimize sgd --weights weights/tmp_6/last.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.005 --s_decay 0.01 --lr_decay 1 --wdir 0.01_6 ',


'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 1.00E-03 --epochs 400 --LR 0.001 --optimize sgd --weights weights/tmp_1/last.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.002 --s_decay 1 --lr_decay 1 --wdir 1_1 ',
'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 2.00E-03 --epochs 400 --LR 0.001 --optimize sgd --weights weights/tmp_2/last.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.002 --s_decay 1 --lr_decay 1 --wdir 1_2 ',
'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 8.00E-04 --epochs 400 --LR 0.001 --optimize sgd --weights weights/tmp_3/last.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.002 --s_decay 1 --lr_decay 1 --wdir 1_3 ',
'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 1.00E-03 --epochs 300 --LR 0.001 --optimize sgd --weights weights/tmp_4/last.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.005 --s_decay 1 --lr_decay 1 --wdir 1_4 ',
'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 2.00E-03 --epochs 300 --LR 0.001 --optimize sgd --weights weights/tmp_5/last.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.005 --s_decay 1 --lr_decay 1 --wdir 1_5 ',
'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 8.00E-04 --epochs 300 --LR 0.001 --optimize sgd --weights weights/tmp_6/last.weights --save_interval 20 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True --s_end 0.005 --s_decay 1 --lr_decay 1 --wdir 1_6 ',

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
