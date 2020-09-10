#-*-coding:utf-8-*-

cmds = [

'python train.py --type spp --activation swish --batch-size 4 --freeze False --epochs 150 --LR 0.00025 --optimize adam --weights weights/darknet53.conv.74 --save_interval 10 --multi-scale True --img_size 416 --rect False --data data/fish/fish.data --expFolder fish	--expID 1134',


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
