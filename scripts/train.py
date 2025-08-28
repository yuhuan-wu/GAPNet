import os
import shutil
import random
import cv2
import torch
from models import model as net
import numpy as np
import transforms as myTransforms
from dataset import Dataset
# from parallel import DataParallelModel, DataParallelCriterion
import time
from argparse import ArgumentParser
from saleval import SalEval
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
# from torch.nn.parallel import gather
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def BCEDiceLoss(inputs, targets, ignore_index=False):
    bce = CrossEntropyLoss(inputs, targets, ignore_index)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    # print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
    return bce + 1 - dice


def CrossEntropyLoss(inputs, targets, ignore_index=False):
    index = [i for i in range(targets.size()[0]) if not torch.all(targets[i] == 10)]

    targets = targets[index, :, :]
    inputs = inputs[index, :, :]

    if ignore_index:
        assert len(torch.unique(targets)) == 3, torch.unique(targets)
        valid_mask = (targets != 255)
        
        targets_valid = targets.clone()
        targets_valid[targets == 255] = 0
        
        BCE_func = nn.BCELoss(reduction='none')
        bce = BCE_func(inputs, targets_valid)
        
        bce = bce * valid_mask
        
        bce = torch.sum(bce) / (torch.sum(valid_mask) + 1e-8)  # Add small epsilon to avoid division by zero
    else:
        assert len(torch.unique(targets)) == 2, torch.unique(targets)
        bce = F.binary_cross_entropy(inputs, targets)

    if bce < 0:
        print(bce)
        exit()
    else:
        return bce


class CEOLoss(nn.Module):
    def __init__(self, criterion=BCEDiceLoss, ignore_index=False, supervision=0):
        super(CEOLoss, self).__init__()
        self.criterion = criterion
        self.ignore_index = ignore_index
        self.supervision = supervision

    def forward(self, inputs, targets):
        # num_scales = 6 if self.dds else 1
        num_scales = 6
        assert len(inputs.shape) == 4, f"prediction has {len(inputs.shape)} (should be 4) dimensions: C, scale, H, W"
        assert inputs.size()[1] == num_scales, f"prediction has {inputs.size()[1]} (should be {num_scales}) levels of features"
        # assert target.size()[1] == num, f"target has {target.size()[1]} (should be 3) levels of features"

        criterion = self.criterion

        losses = []
        for i in [0, 1, 2, 5]:
            dt = inputs[:, i, :, :]
            gt = targets[:, i, :, :]
            losses.append(criterion(dt, gt, ignore_index=False))

        for i in range(3, 5):
            dt = inputs[:, i, :, :]
            gt = targets[:, i, :, :]
            losses.insert(i, criterion(dt, gt, ignore_index=self.ignore_index))

        if self.supervision == 0:
            loss_overall = losses[:1]
        elif self.supervision == 1:
            for i in range(1, 6):
                losses[i] = criterion(inputs[:, i, :, :], targets[:, 0, :, :], ignore_index=False)
            loss_overall = losses[0:2] + losses[3:6]
        elif self.supervision == 2:
            loss_overall = losses[0:2] + losses[3:5]
        elif self.supervision == 3:
            losses[-1] = criterion(inputs[:, -1, :, :], targets[:, 2, :, :], ignore_index=False)
            loss_overall = losses[0:2] + losses[3:6]
        elif self.supervision == 4:
            losses[-1] = criterion(inputs[:, -1, :, :], targets[:, 5, :, :], ignore_index=False)
            loss_overall = losses[0:2] + losses[3:6]
        elif self.supervision == 5:
            losses[-1] = criterion(inputs[:, -1, :, :], targets[:, 3, :, :], ignore_index=self.ignore_index)
            loss_overall = losses[0:2] + losses[3:6]
        elif self.supervision == 6:
            loss_overall = losses[:1] + losses[2:3] + losses[4:5]
        elif self.supervision == 7:
            loss_overall = losses[:1] + losses[2:5]
        elif self.supervision == 8:
            losses[3] = criterion(inputs[:, 3, :, :], targets[:, 2, :, :], ignore_index=False)
            loss_overall = losses[:1] + losses[3:5]
        elif self.supervision == 9:
            loss_overall = losses[:1] + losses[3:5]
        elif self.supervision == 10:
            loss_overall = []
            for i in [0, 3, 4]:
                dt = inputs[:, i, :, :]
                gt = targets[:, 0, :, :]
                loss_overall.append(criterion(dt, gt, ignore_index=False))

        return sum(loss_overall)/len(loss_overall)*3


@torch.no_grad()
def val(args, val_loader, model, criterion):
    # switch to evaluation mode
    model.eval()
    sal_eval_val = SalEval()
    epoch_loss = []
    total_batches = len(val_loader)

    for iter, (input, target) in enumerate(tqdm(val_loader)):
        start_time = time.time()
        input = input.to(device)
        target = target.to(device)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target).float()

        # run the model
        output = model(input_var)

        time_taken = time.time() - start_time

        epoch_loss.append(0)
        # For validation, we use the original label (not processed)
        sal_eval_val.add_batch(output[:, 0, :, :],  target_var)

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    F_beta, MAE = sal_eval_val.get_metric()

    return average_epoch_loss_val, F_beta, MAE


def train(args, train_loader, model, criterion, optimizer, epoch, max_batches, cur_iter=0):
    # switch to train mode
    model.train()
    # sal_eval_train = SalEval()
    epoch_loss = []
    total_batches = len(train_loader)
    iter_time = 0
    optimizer.zero_grad()
    bar = tqdm(train_loader)
    
    # Timing statistics
    timing_stats = {
        'data_loading': 0.0,
        'lr_adjustment': 0.0,
        'data_to_gpu': 0.0,
        'variable_creation': 0.0,
        'resize_if_needed': 0.0,
        'forward_pass': 0.0,
        'loss_calculation': 0.0,
        'backward_pass': 0.0,
        'optimizer_step': 0.0,
        'other': 0.0,
        'total': 0.0
    }
    
    num_iterations = 0
    
    for iter, (input, target) in enumerate(bar):
        iter_start_time = time.time()
        data_loading_time = iter_start_time
        num_iterations += 1
        
        # Adjust the learning rate
        lr_start_time = time.time()
        timing_stats['data_loading'] += lr_start_time - data_loading_time
        
        lr = adjust_learning_rate(args, optimizer, epoch, iter + cur_iter, max_batches)
        
        # Move data to GPU
        gpu_start_time = time.time()
        timing_stats['lr_adjustment'] += gpu_start_time - lr_start_time
        
        input = input.to(device)
        target = target.to(device)
        
        # Create variables
        var_start_time = time.time()
        timing_stats['data_to_gpu'] += var_start_time - gpu_start_time
        
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target).float()
        
        # Resize if needed
        resize_start_time = time.time()
        timing_stats['variable_creation'] += resize_start_time - var_start_time
        
        if args.ms1:
            resize = np.random.choice([320, 352, 384])
            input_var = F.interpolate(input_var, size=(resize, resize), mode='bilinear', align_corners=False)
            target_var = F.interpolate(target_var.unsqueeze(dim=1), size=(resize, resize), mode='bilinear', align_corners=False).squeeze(dim=1)
        
        # Forward pass
        forward_start_time = time.time()
        timing_stats['resize_if_needed'] += forward_start_time - resize_start_time
        
        output = model(input_var)
        
        # Loss calculation
        loss_start_time = time.time()
        timing_stats['forward_pass'] += loss_start_time - forward_start_time
        
        loss = criterion(output, target_var) / args.iter_size
        
        # Backward pass
        backward_start_time = time.time()
        timing_stats['loss_calculation'] += backward_start_time - loss_start_time
        
        loss.backward()
        
        # Optimizer step if needed
        optim_start_time = time.time()
        timing_stats['backward_pass'] += optim_start_time - backward_start_time
        
        iter_time += 1
        if iter_time % args.iter_size == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Other operations
        other_start_time = time.time()
        timing_stats['optimizer_step'] += other_start_time - optim_start_time
        
        epoch_loss.append(loss.data.item())
        
        # End of iteration
        iter_end_time = time.time()
        timing_stats['other'] += iter_end_time - other_start_time
        timing_stats['total'] += iter_end_time - iter_start_time
        
        # Update progress bar with timing info
        if iter % 10 == 0:
            # Find the slowest operation
            slowest_op = max(timing_stats.items(), key=lambda x: x[1] if x[0] != 'total' else 0)
            if slowest_op[0] != 'total':
                slowest_percentage = (slowest_op[1] / timing_stats['total']) * 100 if timing_stats['total'] > 0 else 0
                bar.set_description(
                    "loss: {:.5f}, lr: {:.8f}, slowest: {} ({:.1f}%)".format(
                        sum(epoch_loss) / len(epoch_loss),
                        lr,
                        slowest_op[0],
                        slowest_percentage
                    )
                )
            else:
                bar.set_description("loss: {:.5f}, lr: {:.8f}".format(sum(epoch_loss) / len(epoch_loss), lr))
    
    
    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    F_beta, MAE = 0, 0  # sal_eval_train.get_metric()

    return average_epoch_loss_train, F_beta, MAE, lr


def adjust_learning_rate(args, optimizer, epoch, iter, max_batches):
    if args.lr_mode == 'step':
        lr = args.lr * (0.1 ** (epoch // args.step_loss))
    elif args.lr_mode == 'poly':
        max_iter = max_batches * args.max_epochs
        lr = args.lr * (1 - iter * 1.0 / max_iter) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))
    if epoch == 0 and iter < 200:  # warm up
        lr = args.lr * 0.99 * (iter + 1) / 200 + 0.01 * args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train_validate_saliency(args):
    # load the model
    # add_dwconv = True if args.add_dwconv else False
    # print(add_dwconv)
    model = net.GAPNet(arch=args.arch, pretrained=True)

    args.savedir = args.savedir + '/'
    # create the directory if not exist
    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    print('copying train.py, train.sh, model.py to snapshots dir')
    shutil.copy('scripts/train.py', args.savedir + 'train.py')
    shutil.copy('scripts/train.sh', args.savedir + 'train.sh')
    os.system("scp -r {} {}".format("scripts", args.savedir))
    os.system("scp -r {} {}".format("models", args.savedir))

    if args.gpu and torch.cuda.device_count() > 1:
        # model = nn.DataParallel(model)
        model = nn.DataParallel(model)

    model = model.to(device)

    total_paramters = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters: ' + str(total_paramters))

    NORMALISE_PARAMS = [np.array([0.406, 0.456, 0.485], dtype=np.float32).reshape((1, 1, 3)),  # MEAN, BGR
                        np.array([0.225, 0.224, 0.229], dtype=np.float32).reshape((1, 1, 3))]  # STD, BGR

    # compose the data with transforms
    trainDataset_main = myTransforms.Compose([
        myTransforms.Normalize(*NORMALISE_PARAMS),
        myTransforms.Scale(args.width, args.height),
        myTransforms.RandomCropResize(int(7./224.*args.width)),
        myTransforms.RandomFlip(),
        # myTransforms.GaussianNoise(),
        myTransforms.ToTensor(BGR=False)
    ])

    trainDataset_scale1 = myTransforms.Compose([
        myTransforms.Normalize(*NORMALISE_PARAMS),
        myTransforms.Scale(320, 320),
        myTransforms.RandomCropResize(int(7./224.*320)),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor(BGR=False)
    ])
    trainDataset_scale2 = myTransforms.Compose([
        myTransforms.Normalize(*NORMALISE_PARAMS),
        myTransforms.Scale(352, 352),
        myTransforms.RandomCropResize(int(7./224.*352)),
        myTransforms.RandomFlip(),
        myTransforms.ToTensor(BGR=False)
    ])

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(*NORMALISE_PARAMS),
        myTransforms.Scale(args.width, args.height),
        myTransforms.ToTensor(BGR=False)
    ])

    val_names = ["DUTS-TE", "DUT-OMRON", "HKU-IS", "ECSSD", "PASCAL-S"]

    trainLoader_main = torch.utils.data.DataLoader(
        Dataset(args.data_dir, 'DUTS-TR', transform=trainDataset_main, process_label=True, ignore_index=args.igi),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    trainLoader_scale1 = torch.utils.data.DataLoader(
        Dataset(args.data_dir, 'DUTS-TR', transform=trainDataset_scale1, process_label=True, ignore_index=args.igi),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    trainLoader_scale2 = torch.utils.data.DataLoader(
        Dataset(args.data_dir, 'DUTS-TR', transform=trainDataset_scale2, process_label=True, ignore_index=args.igi),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

    valLoader = torch.utils.data.DataLoader(
        Dataset(args.data_dir, val_names[0], transform=valDataset, process_label=False),
        batch_size=12, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    valLoader1 = torch.utils.data.DataLoader(
        Dataset(args.data_dir, val_names[1], transform=valDataset, process_label=False),
        batch_size=12, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    valLoader2 = torch.utils.data.DataLoader(
        Dataset(args.data_dir, val_names[2], transform=valDataset, process_label=False),
        batch_size=12, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    valLoader3 = torch.utils.data.DataLoader(
        Dataset(args.data_dir, val_names[3], transform=valDataset, process_label=False),
        batch_size=12, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    valLoader4 = torch.utils.data.DataLoader(
        Dataset(args.data_dir, val_names[4], transform=valDataset, process_label=False),
        batch_size=12, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    # valLoader5 = torch.utils.data.DataLoader(
    #     Dataset(args.data_dir, val_names[5], transform=valDataset),
    #     batch_size=12, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    if args.ms:
        max_batches = len(trainLoader_main) + len(trainLoader_scale1) + len(trainLoader_scale2)
    else:
        max_batches = len(trainLoader_main)
    print('max_batches {}'.format(max_batches))
    cudnn.benchmark = True

    start_epoch = 0

    if args.resume is not None:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            # args.lr = checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    log_file = args.savedir + args.log_file
    if os.path.isfile(log_file):
        logger = open(log_file, 'a')
    else:
        logger = open(log_file, 'w')
    logger.write("\nParameters: %s" % (str(total_paramters)))
    logger.write("\n%s\t\t%s\t%s\t%s\t%s\t%s\tlr" % ('Epoch',
                                                     'Loss(Tr)', 'F_beta (tr)', 'MAE (tr)', 'F_beta (val)', 'MAE (val)'))
    logger.flush()

    normal_parameters = []
    picked_parameters = []
    if args.group_lr:
        # use smaller lr in backbone
        for pname, p in model.named_parameters():
            if 'backbone' in pname:
                picked_parameters.append(p)
                print("lr/10", pname)
            else:
                normal_parameters.append(p)
        optimizer = torch.optim.Adam([
            {
                'params': normal_parameters,
                'lr': args.lr,
                'weight_decay': 1e-4
            },
            {
                'params': picked_parameters,
                'lr': args.lr / 10,
                'weight_decay': 1e-4
            },
        ],
            lr=args.lr,
            betas=(0.9, args.adam_beta2),
            eps=1e-08,
            weight_decay=1e-4)
    else:
        optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, args.adam_beta2), eps=1e-08, weight_decay=1e-4)
    cur_iter = 0

    criteria = CrossEntropyLoss
    if args.bcedice:
        criteria = BCEDiceLoss
        print("use dice loss")

    criteria = CEOLoss(criterion=criteria, ignore_index=args.igi, supervision=args.supervision)

    if args.gpu and torch.cuda.device_count() > 1:
        print("using mutliple gpus")

    epoch_idxes = []
    F_beta_vals = []
    F_beta_val1s = []
    F_beta_val2s = []
    F_beta_val3s = []
    F_beta_val4s = []
    F_beta_val5s = []
    MAE_vals = []
    MAE_val1s = []
    MAE_val2s = []
    MAE_val3s = []
    MAE_val4s = []
    MAE_val5s = []

    for epoch in range(start_epoch, args.max_epochs):
        # train for one epoch
        if args.ms:
            train(args, trainLoader_scale1, model, criteria, optimizer, epoch, max_batches, cur_iter)
            cur_iter += len(trainLoader_scale1)
            torch.cuda.empty_cache()
            train(args, trainLoader_scale2, model, criteria, optimizer, epoch, max_batches, cur_iter)
            cur_iter += len(trainLoader_scale2)
            torch.cuda.empty_cache()

        train(args, trainLoader_main, model, criteria, optimizer, epoch, max_batches, cur_iter)
        cur_iter += len(trainLoader_main)
        torch.cuda.empty_cache()

        # evaluate on validation set
        print("start to evaluate on epoch {}".format(epoch+1))
        import time
        start_time = time.time()
        loss_val, F_beta_val, MAE_val = val(args, valLoader, model, criteria)
        torch.cuda.empty_cache()
        if epoch > args.max_epochs * 0.5:
            loss_val1, F_beta_val1, MAE_val1 = val(args, valLoader1, model, criteria)
            torch.cuda.empty_cache()
            loss_val2, F_beta_val2, MAE_val2 = val(args, valLoader2, model, criteria)
            torch.cuda.empty_cache()
            loss_val3, F_beta_val3, MAE_val3 = val(args, valLoader3, model, criteria)
            torch.cuda.empty_cache()
            loss_val4, F_beta_val4, MAE_val4 = val(args, valLoader4, model, criteria)
            torch.cuda.empty_cache()
            # loss_val5, F_beta_val5, MAE_val5 = val(args, valLoader5, model, criteria)
            F_beta_val5, MAE_val5 = 0, 0
            F_beta_vals.append(F_beta_val)
            F_beta_val1s.append(F_beta_val1)
            F_beta_val2s.append(F_beta_val2)
            F_beta_val3s.append(F_beta_val3)
            F_beta_val4s.append(F_beta_val4)
            F_beta_val5s.append(F_beta_val5)
            MAE_vals.append(MAE_val)
            MAE_val1s.append(MAE_val1)
            MAE_val2s.append(MAE_val2)
            MAE_val3s.append(MAE_val3)
            MAE_val4s.append(MAE_val4)
            MAE_val5s.append(MAE_val5)
            epoch_idxes.append(epoch+1)

        print("elapsed evaluation time: {} hours".format((time.time()-start_time)/3600.0))
        torch.cuda.empty_cache()

        torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss_val': loss_val,
            'iou_val': F_beta_val,
        }, args.savedir + 'checkpoint.pth.tar')

        # save the model also

        if epoch > args.max_epochs * 0.5:
            model_file_name = args.savedir + 'model_' + str(epoch + 1) + '.pth'
            print("saving state dict to {}".format(model_file_name))
            torch.save(model.state_dict(), model_file_name)

        log_str = "\n{} {:.4f} {:.4f}".format(epoch+1, F_beta_val, MAE_val)
        try:
            log_str = log_str + " {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(
                F_beta_val1, MAE_val1, F_beta_val2, MAE_val2, F_beta_val3, MAE_val3, F_beta_val4, MAE_val4, F_beta_val5, MAE_val5)
        except:
            pass
        logger.write(log_str)
        logger.flush()
        # print("Epoch " + str(epoch) + ': Details')
        print("Epoch No. %d: \t Val Loss = %.4f\t MAE Loss = %.4f\t F_beta(val) = %.4f\n"
              % (epoch+1, loss_val, MAE_val, F_beta_val))
        torch.cuda.empty_cache()
    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', default="./data/", help='Data directory')
    parser.add_argument('--width', type=int, default=384, help='Width of RGB image')
    parser.add_argument('--height', type=int, default=384, help='Height of RGB image')
    parser.add_argument('--max_epochs', type=int, default=100, help='Max. number of epochs')
    parser.add_argument('--num_workers', type=int, default=10, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='step', help='Learning rate policy, step or poly')
    parser.add_argument('--savedir', default='./gapnet', help='Directory to save the results')
    parser.add_argument('--resume', default=None, help='Use this checkpoint to continue training')
    parser.add_argument('--log_file', default='trainValLog.txt', help='File that stores the training and validation logs')
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--iter_size', default=1, type=int)
    parser.add_argument('--arch', default='vgg16', type=str)
    parser.add_argument('--ms', default=1, type=int)  # normal multi-scale training
    # hybrid multi-scale training. It has comparable performance with normal multi-scale training in my experiments. But I think hybrid multi-scale training may be a better choice.
    parser.add_argument('--ms1', default=0, type=int)
    parser.add_argument('--adam_beta2', default=0.999, type=float)  # The value of 0.99 can introduce slightly higher performance (0.1%~0.2%)
    parser.add_argument('--bcedice', default=0, type=int)
    parser.add_argument('--group_lr', default=0, type=int)
    parser.add_argument('--gpu_id', default='0, 1', type=str)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--igi', default=0, type=int, help="ignore index")
    parser.add_argument('--supervision', default=8, type=int, help="supervision signals")
    args = parser.parse_args()

    seed = args.seed
    #random.seed(seed)
    #np.random.seed(seed)
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    args.savedir += f'seed{args.seed}-'

    print('Called with args:')
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    if args.gpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # print(torch.cuda.device_count())
    args.batch_size *= torch.cuda.device_count()
    args.num_workers *= torch.cuda.device_count()
    train_validate_saliency(args)
