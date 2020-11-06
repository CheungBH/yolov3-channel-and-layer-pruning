device = "cuda:0"

gray_yolo_cfg = "/media/hkuit164/WD20EJRX/result/best_finetune/gray/SLIM-prune_0.95_keep_0.1/prune_0.95_keep_0.1.cfg"
gray_yolo_weights = "/media/hkuit164/WD20EJRX/result/best_finetune/gray/SLIM-prune_0.95_keep_0.1/best.weights"
black_yolo_cfg = "/media/hkuit164/WD20EJRX/result/best_finetune/black/SLIM-prune_0.93_keep_0.1/prune_0.93_keep_0.1.cfg"
black_yolo_weights = "/media/hkuit164/WD20EJRX/result/best_finetune/black/SLIM-prune_0.93_keep_0.1/best.weights"
rgb_yolo_cfg = ""
rgb_yolo_weights = ""

video_path = "/media/hkuit164/WD20EJRX/RegionDetection/0605_4.mp4"
water_top = 40

RNN_frame_length = 4
RNN_backbone = "TCN"
RNN_class = ["stand", "drown"]
RNN_weight = "weights/RNN/TCN_struct1_2020-07-08-20-02-32.pth"
TCN_single = True

'''
----------------------------------------------------------------------------------------------------------------
'''

# For yolo
confidence = 0.4
num_classes = 80
nms_thresh = 0.33
input_size = 416

# For pose estimation
input_height = 320
input_width = 256
output_height = 80
output_width = 64
fast_inference = True
pose_batch = 80

pose_backbone = "seresnet101"
pose_cls = 17
DUCs = [480, 240]


# For detection
frame_size = (720, 540)
resize_ratio = 0.5
store_size = (frame_size[0]*4, frame_size[1]*2)
show_size = (1560, 720)

black_box_threshold = 0.3
gray_box_threshold = 0.2

