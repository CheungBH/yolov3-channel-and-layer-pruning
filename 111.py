# import cv2
# image = cv2.imread('tmp.jpg', 1)
# rows, cols, channel = image.shape
# affineShrinkTranslationRotation = cv2.getRotationMatrix2D((cols/2, rows/2), 180, 1)
# ShrinkTranslationRotation = cv2.warpAffine(image, affineShrinkTranslationRotation, (cols, rows), borderValue=125)
# cv2.imshow('iii',ShrinkTranslationRotation)
# cv2.waitKey(0)


# python3
# import numpy as np
#
# def py_nms(dets, thresh):
#     """Pure Python NMS baseline."""
#     #x1、y1、x2、y2、以及score赋值
#     x1 = dets[:, 0]
#     y1 = dets[:, 1]
#     x2 = dets[:, 2]
#     y2 = dets[:, 3]
#     scores = dets[:, 4]
#
#     #每一个候选框的面积
#     areas = (x2 - x1 + 1) * (y2 - y1 + 1)
#     #order是按照score降序排序的
#     order = scores.argsort()[::-1]
#
#     keep = []
#     while order.size > 0:
#         i = order[0]
#         keep.append(i)
#         #计算当前概率最大矩形框与其他矩形框的相交框的坐标，会用到numpy的broadcast机制，得到的是向量
#         xx1 = np.maximum(x1[i], x1[order[1:]])
#         yy1 = np.maximum(y1[i], y1[order[1:]])
#         xx2 = np.minimum(x2[i], x2[order[1:]])
#         yy2 = np.minimum(y2[i], y2[order[1:]])
#
#         #计算相交框的面积,注意矩形框不相交时w或h算出来会是负数，用0代替
#         w = np.maximum(0.0, xx2 - xx1 + 1)
#         h = np.maximum(0.0, yy2 - yy1 + 1)
#         inter = w * h
#         #计算重叠度IOU：重叠面积/（面积1+面积2-重叠面积）
#         ovr = inter / (areas[i] + areas[order[1:]] - inter)
#
#         #找到重叠度不高于阈值的矩形框索引
#         inds = np.where(ovr <= thresh)[0]
#         #将order序列更新，由于前面得到的矩形框索引要比矩形框在原order序列中的索引小1，所以要把这个1加回来
#         order = order[inds + 1]
#     return keep
#
# # test
# if __name__ == "__main__":
#     dets = np.array([[30, 20, 230, 200, 1],
#                      [50, 50, 260, 220, 0.9],
#                      [210, 30, 420, 5, 0.8],
#                      [430, 280, 460, 360, 0.7]])
#     thresh = 0.35
#     keep_dets = py_nms(dets, thresh)
#     print(keep_dets)
#     print(dets[keep_dets])



import pandas as pd
import os
from models import *
from test import test
path = 'result/gray/gray_result_sean.csv'
df = pd.read_csv(path)
type= df[df['ID']==2][:]['tpye']
activate= df[df['ID']==2][:]['activate']
img_size = df[df['ID']==2][:]['img_size']
path=''
# name is id
for name in os.listdir(path):
    weight_path = os.path.join(path,name)
    cfg = ''

    if not os.path.exists(os.path.join(weight_path,'best.weight')):
        if  not os.path.exists(os.path.join(weight_path,'best.pt')):
            print('error')
        else:
            convert(cfg=cfg, weights=os.path.exists(os.path.join(weight_path,'best.pt')))
            test(id=name,cfg=cfg,weights=os.path.exists(os.path.join(weight_path,'best.pt')),img_size=img_size)
    else:
        test(id=name,cfg=cfg,weights=os.path.exists(os.path.join(weight_path,'best.pt')),img_size=img_size)



print(df[df['ID']==2][:])
