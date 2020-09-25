#-*-coding:utf-8-*-

cmds = [


    # "python train.py --wdir gray26_sE-3 --cfg cfg/yolov3-1cls.cfg --data data/gray/gray.data --weights weights/best.weights --batch-size 4 --epochs 100 -sr --s 0.001 --prune 1",
    # "python train.py --wdir gray26_s2E-3 --cfg cfg/yolov3-1cls.cfg --data data/gray/gray.data --weights weights/best.weights --batch-size 4 --epochs 100 -sr --s 0.002 --prune 1",
    # "python train.py --wdir gray26_s3E-3 --cfg cfg/yolov3-1cls.cfg --data data/gray/gray.data --weights weights/best.weights --batch-size 4 --epochs 100 -sr --s 0.003 --prune 1",
    "python train_finetune.py --wdir gray26_s4E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/gray/gray.data --weights weights/best.weights --batch-size 4 --epochs 2 -sr --s 0.0004 --prune 1",
    # "python train.py --wdir gray26_s5E-3 --cfg cfg/yolov3-1cls.cfg --data data/gray/gray.data --weights weights/best.weights --batch-size 4 --epochs 100 -sr --s 0.005 --prune 1",
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
