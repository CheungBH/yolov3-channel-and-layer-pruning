#-*-coding:utf-8-*-

cmds = [


    # "python train.py --wdir gray26_sE-3 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/gray/gray.data --weights weights/best.weights --batch-size 16 --epochs 80 -sr --s 0.001 --prune 1",
    "python train_sparse.py --wdir gray26_s1E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/gray/gray.data --weights weights/best.weights --batch-size 16 --epochs 150 -sr --s 0.001 --prune 1",
    "python train_sparse.py --wdir gray26_s2E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/gray/gray.data --weights weights/best.weights --batch-size 16 --epochs 140 -sr --s 0.002 --prune 1",
    "python train_sparse.py --wdir gray26_s3E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/gray/gray.data --weights weights/best.weights --batch-size 16 --epochs 130 -sr --s 0.003 --prune 1",
    "python train_sparse.py --wdir gray26_s4E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/gray/gray.data --weights weights/best.weights --batch-size 16 --epochs 120 -sr --s 0.004 --prune 1",
    "python train_sparse.py --wdir gray26_s5E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/gray/gray.data --weights weights/best.weights --batch-size 16 --epochs 110 -sr --s 0.005 --prune 1",
    "python train_sparse.py --wdir gray26_s6E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/gray/gray.data --weights weights/best.weights --batch-size 16 --epochs 100 -sr --s 0.006 --prune 1",
    "python train_sparse.py --wdir gray26_s7E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/gray/gray.data --weights weights/best.weights --batch-size 16 --epochs 100 -sr --s 0.007 --prune 1",
    "python train_sparse.py --wdir gray26_s8E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/gray/gray.data --weights weights/best.weights --batch-size 16 --epochs 100 -sr --s 0.008 --prune 1",
    "python train_sparse.py --wdir gray26_s9E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/gray/gray.data --weights weights/best.weights --batch-size 16 --epochs 100 -sr --s 0.009 --prune 1",
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
