import os

models_folder = "extract"
# folders = [os.path.join(models_folder, f) for f in os.listdir(models_folder)]

folders = [
    "extract/black_sE-3_all_prune_prune_0.95_keep_0.01_25_shortcut",
    "extract/black_sE-3_all_prune_prune_0.95_keep_0.01_25_shortcut_distilled",
    "extract/black_sE-3_all_prune_prune_0.98_keep_0.01_12_shortcut",
    "extract/black_sE-3_all_prune_prune_0.98_keep_0.01_12_shortcut_distilled",
    "extract/gray_1_225_0.01_all_prune_prune_0.88_keep_0.01_20_shortcut",
    "extract/gray_1_225_0.01_all_prune_prune_0.88_keep_0.01_20_shortcut_distilled",
    "extract/gray_1_225_0.01_all_prune_prune_0.95_keep_0.01_10_shortcut",
    "extract/gray_1_225_0.01_all_prune_prune_0.95_keep_0.01_10_shortcut_distilled"
]

cmds = []
for folder in folders:
    files = [os.path.join(folder, name) for name in os.listdir(folder)]
    for file in files:
        if "cfg" in file:
            cfg = file
        elif "best.pt" in file:
            model = file
        else:
            continue

    if "gray" in folder:
        data = "gray"
    elif "black" or "enhance" in folder:
        data = "enhanced"
    else:
        raise ValueError

    cmd = "python metric.py --cfg {0} --weights {1} --data data/swim_{2}/{2}.data".format(cfg, model, data)
    cmds.append(cmd)

for cmd in cmds:
    os.system(cmd)
