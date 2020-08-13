import csv
import os
with open('train.csv')as f:
    f_csv = csv.DictReader(f)
    for row in f_csv:
        print(row)
        cmd= 'CUDA_VISIBLE_DEVICES={} python train.py --wdir {} --cfg {} ' \
        '--multi-scale --img_size 416 --freeze --data data/ceiling.data --batch-size 32 ' \
        '--weights weights/yolov3-spp-ultralytics.pt --epochs 2'.format(row['GPU'],row['wdir'],row['cfg'])
        os.system(cmd)

