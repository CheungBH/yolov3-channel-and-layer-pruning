import os

cmds = []

prune_folder = {
    # "prune_result/gray_sE-3/best.pt": "gray"
    # "weights/black_s2E-3/best.pt": "enhanced",
    # "weights/black_sE-3/backup290.pt": "enhanced",
    # "weights/gray_s2E-3/best.pt": "gray",
    # "weights/gray_sE-3/best.pt": "gray",
    # "prune_result/gray_s2E-3_0.2_0.01_n290/backup290.pt": "gray"
    # "prune_result/gray_sE-3_225_0.01/best.pt": "gray",
    # "prune_result/gray_2_0.2/best.pt": "gray",
    # "prune_result/black_s2E-3_45_0.01/best.pt": "enhanced",
    # "prune_result/black_s2E-3_225_0.01/best.pt": "enhanced",
    # "prune_result/black_sE-3_225_0.01/best.pt": "enhanced",
    "prune_result/black_sE-3/best.pt": "enhanced"
}

only_metric = True


prune = [0.85, 0.9, 0.92, 0.95]
prune_cmd = ["python prune.py --cfg cfg/yolov3-1cls.cfg --data data/swim_{0}/{0}.data --weights {1} --percent {2} --only_metric {3}".
                 format(v, k, per, only_metric) for k, v in prune_folder.items() for per in prune]

shortcut_p = [0.85, 0.9, 0.92, 0.95]
shortcut_prune_cmd = ["python shortcut_prune.py --cfg cfg/yolov3-1cls.cfg --data data/swim_{0}/{0}.data --weights {1} --percent {2} --only_metric {3}".
                          format(v, k, per, only_metric) for k, v in prune_folder.items() for per in shortcut_p]

layer_num = [8, 12]
layer_prune_cmd = ["python layer_prune.py --cfg cfg/yolov3-1cls.cfg --data data/swim_{0}/{0}.data --weights {1} --shortcuts {2}  --only_metric {3}".
                       format(v, k, num, only_metric) for k, v in prune_folder.items() for num in layer_num]

slim_params = [(0.93, 0.1)]
slim_prune_cmd = ["python slim_prune.py --cfg cfg/yolov3-1cls.cfg --data data/swim_{0}/{0}.data --weights {1} --global_percent {2} --layer_keep {3} --only_metric {4}".
                      format(v, k, param[0], param[1], only_metric) for k ,v in prune_folder.items() for param in slim_params]

all_prune_params = [(10, 0.95, 0.01)]
all_prune_cmd = ["python layer_channel_prune.py --cfg cfg/yolov3-1cls.cfg --data data/swim_{0}/{0}.data --weights {1} --shortcuts {2} --global_percent {3} --layer_keep {4} --only_metric {5}".
                     format(v, k, param[0], param[1], param[2], only_metric) for k, v in prune_folder.items() for param in all_prune_params]

cmds += prune_cmd
cmds += shortcut_prune_cmd
cmds += layer_prune_cmd
cmds += slim_prune_cmd
cmds += all_prune_cmd

for cmd in cmds:
    os.system(cmd)
    # print(cmd)
