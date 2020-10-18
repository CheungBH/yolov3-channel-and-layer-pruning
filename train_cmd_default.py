#-*-coding:utf-8-*-

cmds = [
    'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --s 1E-3 --epochs 300 -sr '
    '--LR 1E-3 --optimize sgd --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True '
    '--img_size 416 --rect True --data data/gray/gray.data  --wdir 4'
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
