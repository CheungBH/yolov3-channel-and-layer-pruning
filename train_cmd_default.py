#-*-coding:utf-8-*-

cmds = [
    'python train_sparse.py --cfg pretrained/26/yolov3-original-1cls-leaky.cfg --batch-size 8 --s 1E-3 --epochs 400 -sr '
    '--LR 1E-3 --optimize sgd --weights weights/sparse/1/last.pt --save_interval 10 --multi-scale True '
    '--img_size 416 --rect True --data data/gray/gray.data  --wdir 1'
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
