folders = [
    "prune_result/black_sE-3/layer_prune/prune_8_shortcut",
    "prune_result/black_sE-3/layer_prune/prune_12_shortcut",
    "prune_result/black_sE-3/layer_prune/prune_15_shortcut",
    "prune_result/black_sE-3/layer_prune/prune_18_shortcut",
    "prune_result/black_sE-3/slim_prune/prune_0.8_keep_0.01",
    "prune_result/black_sE-3/slim_prune/prune_0.9_keep_0.01",
    "prune_result/black_sE-3/slim_prune/prune_0.95_keep_0.01",
    "prune_result/black_sE-3/all_prune/prune_0.9_keep_0.01_10_shortcut",
    "prune_result/black_sE-3/all_prune/prune_0.95_keep_0.01_8_shortcut",
    "prune_result/black_sE-3/all_prune/prune_0.95_keep_0.01_12_shortcut"
]

import os
# cmds = ["python train.py --cfg {0}/{1}.cfg --data data/swim_enhanced/enhanced.data --weights {0}/{1}.weights --epochs 100 --batch-size 32".format(folder, folder.split("/")[-1]) for folder in folders]

cmds = []
for folder in folders:
    path_ls = folder.split("/")
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
    # os.system(cmd)
    print(cmd)
