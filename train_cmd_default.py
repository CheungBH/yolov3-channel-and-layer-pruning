#-*-coding:utf-8-*-

cmds = [
'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 4.00E-04 --epochs 200 --LR 0.00025 --optimize sgd --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 1'
'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 4.00E-04 --epochs 200 --LR 0.00025 --optimize adam --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 2'
'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 2.00E-04 --epochs 200 --LR 0.00025 --optimize sgd --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 3'
'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 2.00E-04 --epochs 200 --LR 0.00025 --optimize adam --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 4'
'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 1.00E-03 --epochs 200 --LR 0.001 --optimize sgd --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 5'
'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 1.00E-03 --epochs 200 --LR 0.001 --optimize adam --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 6'
'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 2.00E-03 --epochs 200 --LR 0.001 --optimize sgd --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 7'
'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 2.00E-03 --epochs 200 --LR 0.001 --optimize adam --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 8'
'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 2.00E-04 --epochs 200 --LR 0.0001 --optimize sgd --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 9'
'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 2.00E-04 --epochs 200 --LR 0.0001 --optimize adam --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 10'
'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 1.00E-04 --epochs 200 --LR 0.0001 --optimize sgd --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 11'
'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 1.00E-04 --epochs 200 --LR 0.0001 --optimize adam --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 12'

'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 4.00E-04 --epochs 200 --LR 0.00025 --optimize sgd --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 14'
'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 4.00E-04 --epochs 200 --LR 0.00025 --optimize adam --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 15'
'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 2.00E-04 --epochs 200 --LR 0.00025 --optimize sgd --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 16'
'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 2.00E-04 --epochs 200 --LR 0.00025 --optimize adam --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 17'
'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 1.00E-03 --epochs 200 --LR 0.001 --optimize sgd --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 18'
'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 1.00E-03 --epochs 200 --LR 0.001 --optimize adam --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 19'
'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 2.00E-03 --epochs 200 --LR 0.001 --optimize sgd --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 20'
'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 2.00E-03 --epochs 200 --LR 0.001 --optimize adam --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 21'
'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 2.00E-04 --epochs 200 --LR 0.0001 --optimize sgd --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 22'
'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 2.00E-04 --epochs 200 --LR 0.0001 --optimize adam --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 23'
'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 1.00E-04 --epochs 200 --LR 0.0001 --optimize sgd --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 24'
'python train_sparse.py --cfg pretrained/prev/yolov3-spp-1cls.cfg --batch-size 8 --freeze False --s 1.00E-04 --epochs 200 --LR 0.0001 --optimize adam --weights pretrained/prev/135_608_best.weights --save_interval 10 --multi-scale True --img_size 416 --rect True --data data/gray/gray.data --sr True 	--wdir 25'
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
        cmd = cmd.replace('--sr True', '--sr')
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
