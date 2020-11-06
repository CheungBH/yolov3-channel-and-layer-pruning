#-*-coding:utf-8-*-

cmds = [
    "python train_sparse.py --wdir test_black2_s1E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/black/black.data --weights weights/black_2/best.weights --batch-size 4 --epochs 100 -sr --s 0.0001 --prune 1",
    # "python train_sparse.py --wdir black2_s2E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/black/black.data --weights weights/black_2/best.weights --batch-size 4 --epochs 100 -sr --s 0.0002 --prune 1",
    # "python train_sparse.py --wdir black2_s3E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/black/black.data --weights weights/black_2/best.weights --batch-size 4 --epochs 100 -sr --s 0.0003 --prune 1",
    # "python train_sparse.py --wdir black2_s4E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/black/black.data --weights weights/black_2/best.weights --batch-size 4 --epochs 100 -sr --s 0.0004 --prune 1",
    # "python train_sparse.py --wdir black2_s5E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/black/black.data --weights weights/black_2/best.weights --batch-size 4 --epochs 100 -sr --s 0.0005 --prune 1",
    # "python train_sparse.py --wdir black2_s6E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/black/black.data --weights weights/black_2/best.weights --batch-size 4 --epochs 100 -sr --s 0.0006 --prune 1",
    # "python train_sparse.py --wdir black2_s7E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/black/black.data --weights weights/black_2/best.weights --batch-size 4 --epochs 100 -sr --s 0.0007 --prune 1",
    # "python train_sparse.py --wdir black2_s8E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/black/black.data --weights weights/black_2/best.weights --batch-size 4 --epochs 100 -sr --s 0.0008 --prune 1",
    # "python train_sparse.py --wdir black2_s9E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/black/black.data --weights weights/black_2/best.weights --batch-size 4 --epochs 100 -sr --s 0.0009 --prune 1",
    #
    # "python train_sparse.py --wdir rgb146_s1E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/rgb/rgb.data --weights weights/rgb_146/best.weights --batch-size 4 --epochs 100 -sr --s 0.0001 --prune 1",
    # "python train_sparse.py --wdir rgb146_s2E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/rgb/rgb.data --weights weights/rgb_146/best.weights --batch-size 4 --epochs 100 -sr --s 0.0002 --prune 1",
    # "python train_sparse.py --wdir rgb146_s3E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/rgb/rgb.data --weights weights/rgb_146/best.weights --batch-size 4 --epochs 100 -sr --s 0.0003 --prune 1",
    # "python train_sparse.py --wdir rgb146_s4E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/rgb/rgb.data --weights weights/rgb_146/best.weights --batch-size 4 --epochs 100 -sr --s 0.0004 --prune 1",
    # "python train_sparse.py --wdir rgb146_s5E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/rgb/rgb.data --weights weights/rgb_146/best.weights --batch-size 4 --epochs 100 -sr --s 0.0005 --prune 1",
    # "python train_sparse.py --wdir rgb146_s6E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/rgb/rgb.data --weights weights/rgb_146/best.weights --batch-size 4 --epochs 100 -sr --s 0.0006 --prune 1",
    # "python train_sparse.py --wdir rgb146_s7E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/rgb/rgb.data --weights weights/rgb_146/best.weights --batch-size 4 --epochs 100 -sr --s 0.0007 --prune 1",
    # "python train_sparse.py --wdir rgb146_s8E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/rgb/rgb.data --weights weights/rgb_146/best.weights --batch-size 4 --epochs 100 -sr --s 0.0008 --prune 1",
    # "python train_sparse.py --wdir rgb146_s9E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/rgb/rgb.data --weights weights/rgb_146/best.weights --batch-size 4 --epochs 100 -sr --s 0.0009 --prune 1",
    #
    # "python train_sparse.py --wdir ceiling2_s1E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/ceiling/ceiling.data --weights weights/ceiling_2/best.weights --batch-size 4 --epochs 100 -sr --s 0.0001 --prune 1",
    # "python train_sparse.py --wdir ceiling2_s2E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/ceiling/ceiling.data --weights weights/ceiling_2/best.weights --batch-size 4 --epochs 100 -sr --s 0.0002 --prune 1",
    # "python train_sparse.py --wdir ceiling2_s3E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/ceiling/ceiling.data --weights weights/ceiling_2/best.weights --batch-size 4 --epochs 100 -sr --s 0.0003 --prune 1",
    # "python train_sparse.py --wdir ceiling2_s4E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/ceiling/ceiling.data --weights weights/ceiling_2/best.weights --batch-size 4 --epochs 100 -sr --s 0.0004 --prune 1",
    # "python train_sparse.py --wdir ceiling2_s5E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/ceiling/ceiling.data --weights weights/ceiling_2/best.weights --batch-size 4 --epochs 100 -sr --s 0.0005 --prune 1",
    # "python train_sparse.py --wdir ceiling2_s6E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/ceiling/ceiling.data --weights weights/ceiling_2/best.weights --batch-size 4 --epochs 100 -sr --s 0.0006 --prune 1",
    # "python train_sparse.py --wdir ceiling2_s7E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/ceiling/ceiling.data --weights weights/ceiling_2/best.weights --batch-size 4 --epochs 100 -sr --s 0.0007 --prune 1",
    # "python train_sparse.py --wdir ceiling2_s8E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/ceiling/ceiling.data --weights weights/ceiling_2/best.weights --batch-size 4 --epochs 100 -sr --s 0.0008 --prune 1",
    # "python train_sparse.py --wdir ceiling2_s9E-4 --cfg cfg/yolov3-original-1cls-leaky.cfg --data data/ceiling/ceiling.data --weights weights/ceiling_2/best.weights --batch-size 4 --epochs 100 -sr --s 0.0009 --prune 1",

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
