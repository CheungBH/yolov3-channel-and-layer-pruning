import os

sparse_type = ["metric"]
p_max, p_min = 95, 50
models = [
    "prune_result/black_s2E-3_45_0.01/best.pt",
    "prune_result/black_origin/best.pt",
    "prune_result/black_s2E-3/best.pt",
    "prune_result/black_s2E-3_225_0.01/best.pt",
    "prune_result/black_sE-3/backup290.pt",
    "prune_result/black_sE-3_225_0.01/best.pt",
    "prune_result/gray_s2E-3/best.pt",
    "prune_result/gray_s2E-3_0.2_0.01/best.pt",
    "prune_result/gray_sE-3/best.pt",
    "prune_result/gray_sE-3_225_0.01/best.pt",
    "prune_result/gray_origin/best.pt",
    "prune_result/black_origin_290/backup290.pt",
    "prune_result/black_s2E-3_225_0.01_n270/backup270.pt",
    "prune_result/gray_s2E-3_0.2_0.01_n290/backup290.pt",
    "prune_result/gray_s2E-3_n290/backup290.pt",
    "prune_result/gray_sE-3_225_0.01_n290/backup290.pt"
]

cmds = []
if "shortcut" in sparse_type:
    cmd = ["python detect_sparse.py --cfg cfg/yolov3-1cls.cfg --weights {} --percent_max {} --percent_min {}".format(model, p_max, p_min)
           for model in models]
    cmds += cmd

if "ordinary" in sparse_type:
    cmd = ["python detect_sparse_shortcut.py --cfg cfg/yolov3-1cls.cfg --weights {} --percent_max {} --percent_min {}".format(model, p_max, p_min)
           for model in models]
    cmds += cmd

if "metric" in sparse_type:
    for model in models:
        if "gray" in model:
            data = "gray"
        elif "black" or "enhance" in model:
            data = "enhanced"
        else:
            raise ValueError
        cmd = "python metric.py --cfg cfg/yolov3-1cls.cfg --weights {0} --data data/swim_{1}/{1}.data".format(model, data)
        cmds.append(cmd)

for c in cmds:
    os.system(c)
