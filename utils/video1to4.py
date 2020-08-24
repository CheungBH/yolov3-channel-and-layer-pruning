import cv2
import os
import numpy as np
from copy import deepcopy

fourcc = cv2.VideoWriter_fourcc(*'XVID')


def cut_image(img, bottom=0, top=0, left=0, right=0):
    height, width = img.shape[0], img.shape[1]
    return np.asarray(img[top: height - bottom, left: width - right])

cnt = 0

folder_name = r"D:\PyCharmProject\opencv\video\ceiling_analysis\0710"
videos = [os.path.join(folder_name, video_n) for video_n in os.listdir(folder_name)]
video_num = 0

for video in videos:
    print(video_num)
    video_num += 1
    cap = cv2.VideoCapture(video)
    height, width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    os.makedirs(video[:-4], exist_ok=True)

    outs = [cv2.VideoWriter(os.path.join(video[:-4], "{}.mp4".format(idx)), fourcc, 18, (int(width/2), int(height/2)))
            for idx in range(4)]

    while True:
        ret, frame = cap.read()
        cnt += 1
        if ret:
            frame1 = cut_image(deepcopy(frame), bottom=int(height/2), right=int(width/2))
            frame0 = cut_image(deepcopy(frame), bottom=int(height/2), left=int(width/2))
            frame3 = cut_image(deepcopy(frame), top=int(height/2), right=int(width/2))
            frame2 = cut_image(deepcopy(frame), top=int(height/2), left=int(width/2))
            #
            # frame0 = cv2.resize(frame0, (540, 360))
            # frame1 = cv2.resize(frame1, (540, 360))
            # frame2 = cv2.resize(frame2, (540, 360))
            # frame3 = cv2.resize(frame3, (540, 360))

            # cv2.imshow("f0", frame0)
            # cv2.imshow("f1", frame1)
            # cv2.imshow("f2", frame2)
            # cv2.imshow("f3", frame3)

            outs[0].write(frame0)
            outs[1].write(frame1)
            outs[2].write(frame2)
            outs[3].write(frame3)

            # cv2.waitKey(2)
        else:
            break

# python prune.py --cfg cfg/yolov3-1cls.cfg --data data/swim_enhanced/enhanced.data --weights weights/black_sE-3/best.pt --percent 0.85

# python shortcut_prune.py --cfg cfg/yolov3-1cls.cfg --data data/swim_gray/gray.data weights/gray_s2E-3/best.pt --percent 0.6

# python slim_prune.py --cfg cfg/yolov3-1cls.cfg --data data/swim_enhanced/enhanced.data --weights weights/black_sE-3/backup290.pt --global_percent 0.8 --layer_keep 0.01

# python layer_prune.py --cfg cfg/yolov3-1cls.cfg --data data/swim_enhanced/enhanced.data --weights weights/black_sE-3/backup290.pt --shortcuts 12

# python layer_channel_prune.py --cfg cfg/yolov3-1cls.cfg --data data/swim_enhanced/enhanced.data --weights weights/black_sE-3/backup290.pt --shortcuts 12 --global_percent 0.8 --layer_keep 0.1
