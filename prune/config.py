import os
models = {

"weights/sparse/gray26_sE-3/last.pt": "cfg/yolov3-original-1cls-leaky.cfg",
"weights/sparse/gray26_s1E-4/last.pt": "cfg/yolov3-original-1cls-leaky.cfg",
"weights/sparse/gray26_s2E-4/last.pt": "cfg/yolov3-original-1cls-leaky.cfg",
"weights/sparse/gray26_s3E-4/last.pt": "cfg/yolov3-original-1cls-leaky.cfg",
"weights/sparse/gray26_s4E-4/last.pt": "cfg/yolov3-original-1cls-leaky.cfg",
"weights/sparse/gray26_s5E-4/last.pt": "cfg/yolov3-original-1cls-leaky.cfg",
"weights/sparse/gray26_s6E-4/last.pt": "cfg/yolov3-original-1cls-leaky.cfg",
"weights/sparse/gray26_s7E-4/last.pt": "cfg/yolov3-original-1cls-leaky.cfg",
"weights/sparse/gray26_s8E-4/last.pt": "cfg/yolov3-original-1cls-leaky.cfg",
"weights/sparse/gray26_s9E-4/last.pt": "cfg/yolov3-original-1cls-leaky.cfg",


}

data = "data/gray/gray.data"

# Sparse option
sparse_type = ["shortcut", "ordinary"]
p_max, p_min = 99, 50

# Prune option
only_metric = False

prune = [0.95,0.96,0.97]
shortcut_p = [0.95,0.96,0.97]
layer_num = [8, 12]
slim_params = [(0.95, 0.1)]
all_prune_params = [(10, 0.95, 0.01), (15, 0.95, 0.01)]

# Finetune option
finetune_folders = [
    os.path.join('prune_result/gray26_s{}E-4-last'.format(j), i)
    for j in range(6,10)
    for i in os.listdir('prune_result/gray26_s{}E-4-last'.format(j))
    if os.path.isdir(os.path.join('prune_result/gray26_s{}E-4-last'.format(j), i))
]

print(finetune_folders)
print(len(finetune_folders))
batch_size = 16
epoch = 100
ms = True

# (Distillation)
t_cfg = "weights/teacher/146/yolov3-original-1cls-leaky.cfg"
t_weight = "weights/teacher/146/best.pt"
