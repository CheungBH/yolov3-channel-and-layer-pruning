from models import *
from utils.utils import *
import numpy as np
from copy import deepcopy
from test import test
from terminaltables import AsciiTable
import time
from utils.prune_utils import *
import argparse
from utils.compute_flops import print_model_param_nums, print_model_param_flops


def write_info(m, metric, string):
    params = print_model_param_nums(m)
    flops = print_model_param_flops(m)
    if string == "origin":
        f.write(('\n' + '%70s' * 1) % ("{}".format(os.path.join(folder_str, "origin"))))
    else:
        f.write(('\n' + '%70s' * 1) % ("{}".format(folder_str)))
    f.write(('%15s' * 1) % ("{}".format(flops)))
    processed_metric = [round(m, 4) for m in metric[0]]
    inf_time, _ = obtain_avg_forward_time(random_input, m)
    inf_time = round(inf_time, 4)
    f.write(('%10s' * 9) % (
        "{}".format(inf_time), "{}".format(params), "{}".format(processed_metric[0]), "{}".format(processed_metric[1]),
        "{}".format(processed_metric[2]), "{}".format(processed_metric[3]), "{}".format(processed_metric[4]),
        "{}".format(processed_metric[5]), "{}\n".format(processed_metric[6]),))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='cfg file path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='*.data file path')
    parser.add_argument('--weights', type=str, default='weights/last.pt', help='sparse model weights')
    parser.add_argument('--percent', type=float, default=0.8, help='channel prune percent')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--only_metric', type=bool, default=False, help="whether save cfg and model")
    opt = parser.parse_args()
    print(opt)

    only_metric = opt.only_metric
    percent = opt.percent
    img_size = opt.img_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(opt.cfg, (img_size, img_size)).to(device)
    if opt.weights.endswith('.pt'):
        model.load_state_dict(torch.load(opt.weights)['model'])
    else:
        load_darknet_weights(model, opt.weights)
    print('\nloaded weights from ',opt.weights)

    folder_str = f'prune_{percent}'
    dest_folder = os.path.join("prune_result", opt.weights.split("/")[1], "ordinary_prune", folder_str)
    os.makedirs(dest_folder, exist_ok=True)
    prune_res = open(os.path.join(dest_folder, "prune_res.txt"), "w")

    eval_model = lambda model:test(opt.cfg, opt.data, 
        weights=opt.weights, 
        batch_size=16,
         img_size=img_size,
         iou_thres=0.5,
         conf_thres=0.1,
         nms_thres=0.5,
         save_json=False,
         model=model)
    obtain_num_parameters = lambda model:sum([param.nelement() for param in model.parameters()])

    if only_metric:
        folder_name = "/".join(opt.weights.split("/")[:-1])
        res_file = os.path.join(folder_name, "prune_result.txt")
        if not os.path.exists(res_file):
            f = open(res_file, "w")
            f.write(('\n' + '%70s' * 1) % "Model")
            f.write(('%15s' * 1) % "FLOPs")
            f.write(('%10s' * 9) % ("Time", "Params", "P", "R", "mAP", "F1", "test_GIoU", "test_obj", "test_cls\n"))
            with torch.no_grad():
                origin_metric = eval_model(model)
            write_info(model, origin_metric, "origin")
        else:
            f = open(res_file, "a+")

    print("\nlet's test the original model first:")
    with torch.no_grad():
        origin_model_metric = eval_model(model)

    origin_nparameters = obtain_num_parameters(model)

    CBL_idx, Conv_idx, prune_idx= parse_module_defs(model.module_defs)

    bn_weights = gather_bn_weights(model.module_list, prune_idx)

    sorted_bn = torch.sort(bn_weights)[0]

    # 避免剪掉所有channel的最高阈值(每个BN层的gamma的最大值的最小值即为阈值上限)
    highest_thre = []
    for idx in prune_idx:
        highest_thre.append(model.module_list[idx][1].weight.data.abs().max().item())
    highest_thre = min(highest_thre)

    # 找到highest_thre对应的下标对应的百分比
    percent_limit = (sorted_bn==highest_thre).nonzero().item()/len(bn_weights)

    print(f'Suggested Gamma threshold should be less than {highest_thre:.4f}.')
    print(f'The corresponding prune ratio is {percent_limit:.3f}, but you can set higher.')
    print(f'Suggested Gamma threshold should be less than {highest_thre:.4f}.', file=prune_res)
    print(f'The corresponding prune ratio is {percent_limit:.3f}, but you can set higher.', file=prune_res)

    #%%
    def prune_and_eval(model, sorted_bn, percent=.0):
        model_copy = deepcopy(model)
        thre_index = int(len(sorted_bn) * percent)
        thre = sorted_bn[thre_index]

        print(f'Gamma value that less than {thre:.4f} are set to zero!')
        print(f'Gamma value that less than {thre:.4f} are set to zero!', file=prune_res)

        remain_num = 0
        for idx in prune_idx:

            bn_module = model_copy.module_list[idx][1]

            mask = obtain_bn_mask(bn_module, thre)

            remain_num += int(mask.sum())
            bn_module.weight.data.mul_(mask)
        print("let's test the current model!")
        with torch.no_grad():
            mAP = eval_model(model_copy)[0][2]

        print(f'Number of channels has been reduced from {len(sorted_bn)} to {remain_num}')
        print(f'Prune ratio: {1-remain_num/len(sorted_bn):.3f}')
        print(f"mAP of the 'pruned' model is {mAP:.4f}")

        print(f'Number of channels has been reduced from {len(sorted_bn)} to {remain_num}', file=prune_res)
        print(f'Prune ratio: {1-remain_num/len(sorted_bn):.3f}', file=prune_res)
        print(f"mAP of the 'pruned' model is {mAP:.4f}", file=prune_res)

        return thre

    print('the required prune percent is', percent)
    print('the required prune percent is', percent, file=prune_res)
    threshold = prune_and_eval(model, sorted_bn, percent)
    #%%
    def obtain_filters_mask(model, thre, CBL_idx, prune_idx):

        pruned = 0
        total = 0
        num_filters = []
        filters_mask = []
        for idx in CBL_idx:
            bn_module = model.module_list[idx][1]
            if idx in prune_idx:

                mask = obtain_bn_mask(bn_module, thre).cpu().numpy()
                remain = int(mask.sum())
                pruned = pruned + mask.shape[0] - remain

                if remain == 0:
                    # print("Channels would be all pruned!")
                    # raise Exception
                    max_value = bn_module.weight.data.abs().max()
                    mask = obtain_bn_mask(bn_module, max_value).cpu().numpy()
                    remain = int(mask.sum())
                    pruned = pruned + mask.shape[0] - remain

                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                      f'remaining channel: {remain:>4d}')
                print(f'layer index: {idx:>3d} \t total channel: {mask.shape[0]:>4d} \t '
                      f'remaining channel: {remain:>4d}', file=prune_res)
            else:
                mask = np.ones(bn_module.weight.data.shape)
                remain = mask.shape[0]

            total += mask.shape[0]
            num_filters.append(remain)
            filters_mask.append(mask.copy())

        prune_ratio = pruned / total
        print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}')
        print(f'Prune channels: {pruned}\tPrune ratio: {prune_ratio:.3f}', file=prune_res)

        return num_filters, filters_mask

    num_filters, filters_mask = obtain_filters_mask(model, threshold, CBL_idx, prune_idx)

    #%%
    CBLidx2mask = {idx: mask.astype('float32') for idx, mask in zip(CBL_idx, filters_mask)}

    pruned_model = prune_model_keep_size2(model, CBL_idx, CBL_idx, CBLidx2mask)

    print("\nnow prune the model but keep size,(actually add offset of BN beta to next layer), let's see how the mAP goes")
    with torch.no_grad():
        eval_model(pruned_model)


    #%%
    compact_module_defs = deepcopy(model.module_defs)
    for idx, num in zip(CBL_idx, num_filters):
        assert compact_module_defs[idx]['type'] == 'convolutional'
        compact_module_defs[idx]['filters'] = str(num)

    #%%
    compact_model = Darknet([model.hyperparams.copy()] + compact_module_defs, (img_size, img_size)).to(device)
    compact_nparameters = obtain_num_parameters(compact_model)

    init_weights_from_loose_model(compact_model, pruned_model, CBL_idx, Conv_idx, CBLidx2mask)

    #%%
    random_input = torch.rand((1, 3, img_size, img_size)).to(device)

    def obtain_avg_forward_time(input, model, repeat=200):

        model.eval()
        start = time.time()
        with torch.no_grad():
            for i in range(repeat):
                output = model(input)[0]
        avg_infer_time = (time.time() - start) / repeat

        return avg_infer_time, output

    print('\ntesting avg forward time...')
    pruned_forward_time, pruned_output = obtain_avg_forward_time(random_input, pruned_model)
    compact_forward_time, compact_output = obtain_avg_forward_time(random_input, compact_model)

    diff = (pruned_output-compact_output).abs().gt(0.001).sum().item()
    if diff > 0:
        print('Something wrong with the pruned model!')

    #%%
    # 在测试集上测试剪枝后的模型, 并统计模型的参数数量
    print('testing the mAP of final pruned model')
    with torch.no_grad():
        compact_model_metric = eval_model(compact_model)


    #%%
    # 比较剪枝前后参数数量的变化、指标性能的变化
    metric_table = [
        ["Metric", "Before", "After"],
        ["mAP", f'{origin_model_metric[0][2]:.6f}', f'{compact_model_metric[0][2]:.6f}'],
        ["Parameters", f"{origin_nparameters}", f"{compact_nparameters}"],
        ["Inference", f'{pruned_forward_time:.4f}', f'{compact_forward_time:.4f}']
    ]
    print(AsciiTable(metric_table).table)
    print(AsciiTable(metric_table).table, file=open(os.path.join(dest_folder, "metric.txt"), "w"))

    #%%
    # 生成剪枝后的cfg文件并保存模型
    if not only_metric:
        cfg_name = os.path.join(dest_folder, folder_str + ".cfg")
        pruned_cfg_file = write_cfg(cfg_name, [model.hyperparams.copy()] + compact_module_defs)
        print(f'Config file has been saved: {pruned_cfg_file}')

        model_name = os.path.join(dest_folder, folder_str + ".weights")
        save_weights(compact_model, model_name)
        print(f'Compact model has been saved: {model_name}')
    else:
        write_info(compact_model, compact_model_metric, "pruned")
        f.close()

