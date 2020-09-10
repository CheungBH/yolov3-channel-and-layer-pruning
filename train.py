import argparse
import traceback
import csv
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_schedule
from cfg import config
import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *
from utils.prune_utils import *
from utils.compute_flops import print_model_param_flops, print_model_param_nums
from torch.utils.tensorboard import SummaryWriter
from utils.pytorchtools import EarlyStopping
mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    mixed_precision = False  # not installed

titles = ['GIoU', 'Objectness', 'Classification', 'Train loss',
          'Precision', 'Recall', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification', 'soft_loss']
# import netron
# netron.start('cfg/yolov3-tiny-1cls-leaky.cfg')

# Overwrite hyp with hyp*`.txt (optional)
f = glob.glob('hyp*.txt')
if f:
    for k, v in zip(config.hyp.keys(), np.loadtxt(f[0])):
        config.hyp[k] = v

def train():
    cfg = os.path.join('cfg','yolov3-'+opt.type+'-'+'1cls'+'-'+opt.activation+'.cfg')
    data = opt.data
    img_size = opt.img_size
    epochs = 1 if opt.prebias else opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = opt.accumulate  # effective bs = batch_size * accumulate = 16 * 4 = 64
    weights = opt.weights  # initial training weights
    convert_weight = True
    config.hyp['lr0'] = opt.LR
    # i.e ./gray/spp/1(./opt.expFolder/opt.type/opt.expID)
    # result_dir = os.path.join((opt.expFolder+'_'+opt.type),opt.expID)
    result_dir = os.path.join(opt.expFolder , opt.expID)
    train_dir = os.path.join('result', result_dir) + os.sep  # train result dir
    weight_dir = os.path.join('weights', result_dir) + os.sep  # weights dir

    if "last" not in weights or not opt.resume:
        if 'test' not in weight_dir or not os.path.exists(weight_dir):
            os.makedirs(weight_dir)
    last = weight_dir + 'last.pt'
    best = weight_dir + 'best.pt'
    os.makedirs(train_dir, exist_ok=True)
    results_txt = os.path.join(train_dir, 'results.txt')#results.txt in weights_folder

    opt.weights = last if opt.resume else opt.weights# if resume use the last

    tb_writer = SummaryWriter('tensorboard/{}/{}'.format(opt.expFolder,opt.expID))
    if 'pw' not in opt.arc:  # remove BCELoss positive weights
        config.hyp['cls_pw'] = 1.
        config.hyp['obj_pw'] = 1.
    early_stoping_giou = EarlyStopping(patience=config.patience,verbose=True)
    early_stoping_obj = EarlyStopping(patience=config.patience, verbose=True)
    # Initialize
    init_seeds()
    multi_scale = opt.multi_scale

    if multi_scale:
        img_sz_min = round(img_size / 32 / 1.5) + 1
        img_sz_max = round(img_size / 32 * 1.5) - 1
        img_size = img_sz_max * 32  # initiate with maximum multi_scale size
        print('Using multi-scale %g - %g' % (img_sz_min * 32, img_size))

    # Configure run
    data_dict = parse_data_cfg(data)
    train_path = data_dict['train']
    nc = int(data_dict['classes'])  # number of classes
    # Remove previous results
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_txt):
        os.remove(f)
    # Initialize model
    model = Darknet(cfg, (img_size, img_size), arc=opt.arc).to(device)
    # Optimizer
    pg0, pg1 = [], []  # optimizer parameter groups
    for k, v in dict(model.named_parameters()).items():
        if 'Conv2d.weight' in k:
            pg1 += [v]  # parameter group 1 (apply weight_decay)
        else:
            pg0 += [v]  # parameter group 0

    flops = print_model_param_flops(model)
    print("The FLOPs of current model is {}".format(flops))
    params = print_model_param_nums(model)
    print("The params of current model is {}".format(params))
    infer_time=get_inference_time(model)
    print("Infer time of current model is {}".format(infer_time))

    if opt.optimize == 'adam':#将网络数数放到优化器
        optimizer = optim.Adam(pg0, lr=config.hyp['lr0'])
        # optimizer = AdaBound(pg0, lr=config.hyp['lr0'], final_lr=0.1)
    elif opt.optimize == 'sgd':
        optimizer = optim.SGD(pg0, lr=config.hyp['lr0'], momentum=config.hyp['momentum'], nesterov=True)
    # elif opt.optimize == 'rmsprop':
    #     optimizer = optim.
    optimizer.add_param_group({'params': pg1, 'weight_decay': config.hyp['weight_decay']})  # add pg1 with weight_decay
    del pg0, pg1

    cutoff = -1  # backbone reaches to cutoff layer
    start_epoch = 0
    best_fitness = 0.
    attempt_download(weights)
    # if the folder is not exist, create it
    with open(results_txt, 'a+') as file:
        file.write(('%10s' * 18) % (
            'Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'soft', 'rratio', 'targets', 'img_size','lr',
            "P", "R", "mAP", "F1", "test_GIoU", "test_obj", "test_cls\n"))

    if weights.endswith('.pt'):  # pytorch format
        # possible weights are 'last.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        if opt.bucket:
            os.system('gsutil cp gs://%s/last.pt %s' % (opt.bucket, last))  # download from bucket
        chkpt = torch.load(weights, map_location=device)

        # load model
        chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(chkpt['model'], strict=False)
        print('loaded weights from', weights, '\n')

        # load optimizer
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']

        # load results
        if chkpt.get('training_results') is not None:
            with open(results_txt, 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt
        start_epoch = chkpt['epoch'] + 1
        del chkpt

    elif len(weights) > 0:  # darknet format
        # possible weights are 'yolov3.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        cutoff = load_darknet_weights(model, weights)
        print('loaded weights from', weights, '\n')

    if opt.freeze and opt.type in ['spp','normal']:
        # spp , normal
        for k, p in model.named_parameters():
            # if 'BatchNorm2d' in k and int(k.split('.')[1]) > 33: #open bn
            if 'BatchNorm2d' in k:
                p.requires_grad = False
            elif int(k.split('.')[1]) < 33:
                p.requires_grad = False
            else:
                p.requires_grad = True
    elif opt.type == 'tiny':
        #tiny
        for k, p in model.named_parameters():
            # if 'BatchNorm2d' in k and int(k.split('.')[1]) > 33: #open bn
            if 'BatchNorm2d' in k:
                p.requires_grad = False
            elif int(k.split('.')[1]) < 9:
                p.requires_grad = False
            else:
                p.requires_grad = True

    if opt.transfer or opt.prebias:  # transfer learning edge (yolo) layers
        nf = int(model.module_defs[model.yolo_layers[0] - 1]['filters'])  # yolo layer size (i.e. 255)

        if opt.prebias:
            for p in optimizer.param_groups:
                # lower param count allows more aggressive training settings: i.e. SGD ~0.1 lr0, ~0.9 momentum
                p['lr'] *= 100  # lr gain
                if p.get('momentum') is not None:  # for SGD but not Adam
                    p['momentum'] *= 0.9

        for p in model.parameters():
            if opt.prebias and p.numel() == nf:  # train (yolo biases)
                p.requires_grad = True
            elif opt.transfer and p.shape[0] == nf:  # train (yolo biases+weights)
                p.requires_grad = True
            else:  # freeze layer
                p.requires_grad = False

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=1)
    # Initialize distributed training
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[0,1,2])
        model.module_list = model.module.module_list
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # Dataset
    dataset = LoadImagesAndLabels(train_path,
                                  img_size,
                                  batch_size,
                                  augment=True,
                                  hyp=config.hyp,  # augmentation hyperparameters
                                  rect=opt.rect,  # rectangular training
                                  image_weights=opt.img_weights,
                                  cache_labels=True if epochs > 10 else False,
                                  cache_images=False if opt.prebias else opt.cache_images)

    # Dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=5,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Start training
    model.nc = nc  # attach number of classes to model
    model.arc = opt.arc  # attach yolo architecture
    model.hyp = config.hyp  # attach hyperparameters to model
    # model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights
    torch_utils.model_info(model, report='summary')  # 'full' or 'summary'
    nb = len(dataloader)
    maps = np.zeros(nc)  # mAP per class
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    best_result = [float("inf"), float("inf"), 0, float("inf"), 0, 0, 0, 0, float("inf"), float("inf"), 0, float("inf")]

    print('Starting %s for %g epochs...' % ('prebias' if opt.prebias else 'training', epochs))
    train_loss_ls,val_loss_ls,prmf_ls  = [],[],[]
    stop = False
    decay = 0
    decay_epoch = []
    patience_decay = config.patience_decay
    final_epoch = 0
    best_epoch = 0
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        print(('\n' + '%10s' * 11) % ('Epoch', 'gpu_mem', 'GIoU', 'obj', 'cls', 'total', 'soft', 'rratio', 'targets', 'img_size','lr'))
        # Freeze backbone at epoch 0, unfreeze at epoch 1 (optional)
        freeze_backbone = True
        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if int(name.split('.')[1]) < cutoff:  # if layer < 75
                    p.requires_grad = False if epoch == 0 else True

        # Update image weights (optional)
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(4).to(device)  # mean losses
        msoft_target = torch.zeros(1).to(device)
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        sr_flag = get_sr_flag(epoch, opt.sr)
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            # 调整学习率，进行warm up和学习率衰减
            if epoch < config.warm_up:
                lr = adjust_learning_rate(optimizer, epoch, ni, nb,config.hyp)
            imgs = imgs.to(device)
            targets = targets.to(device)
            # Multi-Scale training
            if multi_scale:
                if ni / accumulate % 10 == 0:  #  adjust (67% - 150%) every 10 batches
                    img_size = random.randrange(img_sz_min, img_sz_max + 1) * 32
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / 32.) * 32 for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            #Plot images with bounding boxes make sure the labels are correct
            if ni == 0:
                fname = 'train_batch%g.jpg' % i
                plot_images(imgs=imgs, targets=targets, paths=paths, fname=fname)
                if tb_writer:
                    tb_writer.add_image(fname, cv2.imread(fname)[:, :, ::-1], dataformats='HWC')

            # Run model
            pred = model(imgs)

            # Compute loss
            loss, loss_items = compute_loss(pred, targets, model)
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results
            soft_target = 0
            reg_ratio = 0  #表示有多少target的回归是不如老师的，这时学生会跟gt再学习

            # Compute gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Accumulate gradient for x batches before optimizing
            if ni % accumulate == 0:
                optimizer.step()#更新梯度
                optimizer.zero_grad()

            # Print batch results
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            msoft_target = (msoft_target * i + soft_target) / (i + 1)
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 9) % (
                '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, msoft_target, reg_ratio, len(targets), img_size,lr)
            pbar.set_description(s)

            # end batch ------------------------------------------------------------------------------------------------

        # Process epoch results
        final_epoch = epoch + 1 == epochs
        if opt.prebias:
            print_model_biases(model)
        else:
            # Calculate mAP (always test final epoch, skip first 10 if opt.nosave)
            if not (opt.notest or (opt.nosave and epoch < 10)) or final_epoch:
                with torch.no_grad():
                    r,results, maps = test.test(cfg,
                                              data,
                                              batch_size=batch_size,
                                              img_size=opt.img_size,
                                              model=model,
                                              conf_thres=0.001 if final_epoch and epoch > 0 else 0.1,  # 0.1 for speed
                                              save_json=final_epoch and epoch > 0 and 'coco.data' in data,
                                              writer=tb_writer,
                                              write_txt = False)

        # Write epoch resultscfg
        with open(results_txt, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

        # train csv
        # train_result = os.path.join('result', result_dir) +os.sep
        exist = os.path.exists(train_dir+ 'train_csv.csv')
        # os.makedirs(train_dir, exist_ok=True)
        with open(train_dir + 'train_csv.csv', 'a+', newline='')as f:
            f_csv = csv.writer(f)
            if not exist:
                title = ['Epoch', 'GIoU', 'obj', 'cls', 'total',
                         'lr',"P", "R", "mAP", "F1", "test_GIoU", "test_obj", "test_cls"]
                f_csv.writerow(title)
            info_str =[epoch,*mloss.cpu().numpy(),lr ]
            info_str.extend(list(results))
            f_csv.writerow(info_str)


            train_loss_ls.append([*mloss.cpu().numpy()])
            val_loss_ls.append(list(results)[-3:])
            prmf_ls.append(list(results)[0:4])

        # open tensorboard
        with open(os.path.join(train_dir, "tb.py"), "w") as pyfile:
            pyfile.write("import os\n")
            pyfile.write("os.system('conda init bash')\n")
            pyfile.write("os.system('conda activate yolo')\n")
            pyfile.write("os.system('tensorboard --logdir=../../../../tensorboard/{}/{}')"
                         .format(opt.expFolder+'_'+opt.type, opt.expID))

        # Write Tensorboard results
        if tb_writer:
            x = list(mloss) + list(results) + [msoft_target]
            for xi, title in zip(x, titles):
                tb_writer.add_scalar(title, xi, epoch)
            tb_writer.add_scalar('lr', lr, epoch)
            best_result = update_result(best_result, x)
            tb_writer.add_image("result of epoch {}".format(epoch), cv2.imread("tmp.jpg")[:, :, ::-1],
                                dataformats='HWC')

        # Update best mAP
        fitness, P = results[2], results[0]  # mAP
        if fitness > best_fitness and P > 0.5:
            best_fitness = fitness
            best_epoch = epoch

        #Early stoping for Giou
        if epoch == config.warm_up:
            lr = opt.LR
        if epoch > config.warm_up:
            early_stoping_giou(list(results)[-3],list(results)[-2])#valGiou
            if early_stoping_giou.early_stop :
                optimizer, lr = lr_decay(optimizer, lr)
                decay += 1
                if decay > opt.lr_decay_time:
                    stop = True
                else:
                    decay_epoch.append(epoch)
                    early_stoping_giou.reset(int(config.patience * patience_decay[decay]))

        # Save training results
        save = (not opt.nosave) or (final_epoch and not opt.evolve) or opt.prebias
        # save = True
        if save:
            with open(results_txt, 'r') as f:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                         'best_fitness': best_fitness,
                         'training_results': f.read(),
                         'model': model.module.state_dict() if type(
                             model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                         'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last checkpoint
            torch.save(chkpt, last)
            if convert_weight:
                convert(cfg=cfg, weights=weight_dir + 'last.pt')
                os.remove(weight_dir + 'last.pt')

            if opt.bucket and not opt.prebias:
                os.system('gsutil cp %s gs://%s' % (last, opt.bucket))  # upload to bucket

            # Save best checkpoint
            if best_fitness == fitness:
                torch.save(chkpt, best)
                if convert_weight:
                    convert(cfg=cfg, weights=weight_dir + 'best.pt' )
                    os.remove(weight_dir + 'best.pt')

            # Save backup every 10 epochs (optional)
            if  epoch> config.epoch and epoch % opt.save_interval == 0:
                torch.save(chkpt, weight_dir + 'backup%g.pt' % epoch)
                if convert_weight:
                    convert(cfg=cfg, weights=weight_dir + 'backup%g.pt' % epoch)
                    os.remove(weight_dir + 'backup%g.pt' % epoch)
            # Delete checkpoint
            del chkpt
        if stop:
            final_epoch = epoch+1
            print("Training finished at epoch {}".format(epoch))
            break
        # end epoch ----------------------------------------------------------------------------------------------------



    # end training
    train_time = (time.time() - t0) / 3600
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1,train_time))
    #draw graph
    draw_graph(epoch - start_epoch + 1, train_loss_ls, val_loss_ls, prmf_ls, train_dir)

    whole_result = os.path.join('result',opt.expFolder,"{}_result_{}.csv".format(opt.expFolder,config.computer))
    exist = os.path.exists(whole_result)

    with open(whole_result, "a+") as f:
        f_csv = csv.writer(f)
        if not exist:
            title = [
                     'ID','tpye','activation','batch_size','optimize','freeze','epoch_num','LR', 'weights','multi-scale',
                     'img_size','rect','data','model_location', 'folder_name','parameter','flops','infer_time',
                'train_GIoU', 'train_obj','train_cls',
                     'total',"P", "R", "mAP", "F1", "val_GIoU", "val_obj", "val_cls",'train_time',
                     'final_epoch','best_epoch','decay_1','decay_2'
                     ]
            f_csv.writerow(title)
        infostr = [
                   opt.expID,opt.type, opt.activation,opt.batch_size, opt.optimize,opt.freeze,opt.epochs,
                   opt.LR,opt.weights,multi_scale,opt.img_size,opt.rect,opt.data,config.computer, train_dir,params,flops,infer_time

        ]

        best_result = res2list(best_result)
        infostr.extend(best_result[:-1])
        infostr.extend([train_time,final_epoch,best_epoch])
        infostr.extend(decay_epoch)
        f_csv.writerow(infostr)

    plot_results(train_dir)  # save as results.png

    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=273)  # 500200 batches at bs 16, 117263 images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=16)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--weight_dir', type=str, default="test")
    parser.add_argument('--accumulate', type=int, default=2, help='batches to accumulate before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-1cls-leaky.cfg', help='cfg file path')
    parser.add_argument('--t_cfg', type=str, default='', help='teacher model cfg file path for knowledge distillation')
    parser.add_argument('--data', type=str, default='data/coco.data', help='*.data file path')
    parser.add_argument('--test_data', type=str, default='data/ceiling.data', help='*.data file path')
    parser.add_argument('--multi-scale', action='store_true', default=True,help='adjust (67% - 150%) img_size every 10 batches')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--transfer', action='store_true', help='transfer learning')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--img-weights', action='store_true', help='select training images by weight')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--weights', type=str, default='', help='initial weights')  # i.e. weights/darknet.53.conv.74
    parser.add_argument('--t_weights', type=str, default='', help='teacher model weights')
    parser.add_argument('--arc', type=str, default='defaultpw', help='yolo architecture')  # defaultpw, uCE, uBCE
    parser.add_argument('--prebias', action='store_true', help='transfer-learn yolo biases prior to training')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--var', type=float, help='debug variable')
    parser.add_argument('--sparsity-regularization', '-sr', dest='sr', action='store_true', help='train with channel sparsity regularization')
    parser.add_argument('--s', type=float, default=0.001, help='scale sparse rate')
    parser.add_argument('--prune', type=int, default=1, help='0:nomal prune 1:other prune ')
    parser.add_argument('--freeze',action='store_true',default=False,  help='freeze layers ')
    # parser.add_argument('--freeze_percent', type=int, default=0.5)
    parser.add_argument('--expID', type=str, default='0', help='model number')
    parser.add_argument('--LR', type=float,default=0.001, help='learning rate')
    parser.add_argument('--type', type=str, default='spp', help='yolo type(spp,normal,tiny)')
    parser.add_argument('--activation', type=str, default='leaky', help='activation function(leaky,swish,mish)')
    parser.add_argument('--expFolder', type=str, default='gray', help='expFloder')
    parser.add_argument('--save_interval', default=1, type=int,help='interval')
    parser.add_argument('--optimize', type=str, default='sgd', help='optimizer(adam,sgd)')
    parser.add_argument('--lr_decay_time', type=int, default=2, help='lr decay time')

    opt = parser.parse_args()

    print(opt)
    device = torch_utils.select_device(opt.device, apex=mixed_precision)

    tb_writer = None
    if not opt.evolve:  # Train normally
        try:
            train()  # train normally
        except  :
            with open('error.txt','a+') as f:
                f.write(opt.expID)
                f.write('\n')
                f.write('----------------------------------------------\n')
                traceback.print_exc(file=f)


    else:  # Evolve hyperparameters (optional)
        opt.notest = True  # only test final epoch
        opt.nosave = True  # only save final checkpoint
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(1):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                x = np.loadtxt('evolve.txt', ndmin=2)
                parent = 'weighted'  # parent selection method: 'single' or 'weighted'
                if parent == 'single' or len(x) == 1:
                    x = x[fitness(x).argmax()]
                elif parent == 'weighted':  # weighted combination
                    n = min(10, x.shape[0])  # number to merge
                    x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                    w = fitness(x) - fitness(x).min()  # weights
                    x = (x[:n] * w.reshape(n, 1)).sum(0) / w.sum()  # new parent
                for i, k in enumerate(config.hyp.keys()):
                    config.hyp[k] = x[i + 7]

                # Mutate
                np.random.seed(int(time.time()))
                s = [.2, .2, .2, .2, .2, .2, .2, .0, .02, .2, .2, .2, .2, .2, .2, .2, .2]  # sigmas
                for i, k in enumerate(config.hyp.keys()):
                    x = (np.random.randn(1) * s[i] + 1) ** 2.0  # plt.hist(x.ravel(), 300)
                    config.hyp[k] *= float(x)  # vary by sigmas

            # Clip to limits
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                config.hyp[k] = np.clip(config.hyp[k], v[0], v[1])

            results = train()

            # Write mutation results
            print_mutation(config.hyp, results, opt.bucket)

            # Plot results
            # plot_evolution_results(config.hyp)
