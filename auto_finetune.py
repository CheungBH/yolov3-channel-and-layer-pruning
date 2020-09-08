folders = [
    # "prune_result/gray_1_225_0.01/all_prune/prune_0.8_keep_0.01_10_shortcut",
    # "prune_result/gray_1_225_0.01/all_prune/prune_0.85_keep_0.01_10_shortcut",
    # "prune_result/gray_1_225_0.01/layer_prune/prune_8_shortcut",
    # "prune_result/gray_1_225_0.01/layer_prune/prune_12_shortcut",
    # "prune_result/gray_1_225_0.01/layer_prune/prune_15_shortcut",
    # "prune_result/gray_1_225_0.01/slim_prune/prune_0.85_keep_0.01",
    # "prune_result/gray_1_225_0.01/slim_prune/prune_0.88_keep_0.01",
    # "prune_result/gray_2_0.2/all_prune/prune_0.85_keep_0.01_10_shortcut",
    # "prune_result/gray_2_0.2/layer_prune/prune_15_shortcut",
    # "prune_result/gray_2_0.2/slim_prune/prune_0.85_keep_0.01"
    # "prune_result/gray_1_225_0.01/layer_prune/prune_20_shortcut",
    # "prune_result/gray_1_225_0.01/slim_prune/prune_0.93_keep_0.1",
    # "prune_result/gray_1_225_0.01/all_prune/prune_0.88_keep_0.01_16_shortcut",
    # "prune_result/gray_1_225_0.01/all_prune/prune_0.9_keep_0.1_15_shortcut"
    "prune_result/gray_1_225_0.01/all_prune/prune_0.95_keep_0.01_10_shortcut",
    "prune_result/gray_1_225_0.01/all_prune/prune_0.88_keep_0.01_20_shortcut",
]

import os
# cmds = ["python train.py --cfg {0}/{1}.cfg --data data/swim_gray/gray.data --weights {0}/{1}.weights --epochs 100 --batch-size 32".format(folder, folder.split("/")[-1]) for folder in folders]

cmds = []
for folder in folders:
    path_ls = folder.split("/")
    wdir = path_ls[1] + "_" + path_ls[2] + "_" + path_ls[3] + "_distilled"
    cmds.append("python train.py --wdir finetune/{2} --cfg {0}/{1}.cfg --data data/swim_gray/gray.data --weights {0}/{1}.weights --epochs 100 --batch-size 32 --t_cfg cfg/yolov3-1cls.cfg --t_weights weights/gray_origin/best.pt".format(folder, folder.split("/")[-1], wdir))
    wdir = path_ls[1] + "_" + path_ls[2] + "_" + path_ls[3]

    if "gray" in folder:
        data = "gray"
    elif "enhanced" or "black" in folder:
        data = "enhanced"
    else:
        raise ValueError
    cmds.append("python train.py --wdir {2} --cfg {0}/{1}.cfg --data data/swim_{3}/{3}.data --weights {0}/{1}.weights "
                "--epochs 100 --batch-size 32".format(folder, folder.split("/")[-1], wdir, data))

for cmd in cmds:
    os.system(cmd)
    # print(cmd)
