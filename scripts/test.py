import shutil
import torch
import cv2
import time
import os
import os.path as osp
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from argparse import ArgumentParser
from collections import OrderedDict
from saleval import SalEval
from models import model as net
from tqdm import tqdm
# from train2 import gt2gt_ms
import random
from fvcore.nn import FlopCountAnalysis


@torch.no_grad()
def test(args, model, image_list, label_list, save_dir):

    mean = [0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229]
    eval = SalEval()

    for idx in tqdm(range(len(image_list))):
        # for idx in tqdm(range(1)):
        image = cv2.imread(image_list[idx])
        label = cv2.imread(label_list[idx], 0)
        label = label / 255

        # resize the image to 1024x512x3 as in previous papers
        img = cv2.resize(image, (args.width, args.height))
        img = img.astype(np.float32) / 255.
        img -= mean
        img /= std

        img = img[:, :, ::-1].copy()
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0)
        img = Variable(img)
        # print(img.size())
        label = torch.from_numpy(label).float().unsqueeze(0)

        img = img.to(device)
        label = label.to(device)

        num_areas = 6 if args.dds else 1
        # full map, edge, center, center+others, edge+others, others
        #areas = [#['_full_map', '_edge', '_center', '_center_other', '_edge_other', '_high_global']
        imgs_out = model(img)
        img_out = imgs_out[:, 0, :, :].unsqueeze(dim=0)
        img_out = F.interpolate(img_out, size=image.shape[:2], mode='bilinear', align_corners=False)
        sal_map = (img_out*255).data.cpu().numpy()[0, 0].astype(np.uint8)
        cv2.imwrite(osp.join(save_dir, osp.basename(image_list[idx])[:-4] +'.png'), sal_map)
        # for i, area in enumerate(areas[0:num_areas]):
        #     img_out = imgs_out[:, i, :, :].unsqueeze(dim=0)
        #     # print(idx, i, torch.max(img_out), torch.min(img_out), torch.mean(img_out))
        #     img_out = F.interpolate(img_out, size=image.shape[:2], mode='bilinear', align_corners=False)
        #     sal_map = (img_out*255).data.cpu().numpy()[0, 0].astype(np.uint8)
        #     # gt_map = (label.unsqueeze(dim=0)*255).data.cpu().numpy()[0, 0].astype(np.uint8)
        #     # print(np.min(gt_map), np.max(gt_map), gt_map.shape())
        #     # print(osp.basename(image_list[idx])[-8:-4])
        #     if osp.basename(image_list[idx])[-12:-4] in ['00000003', '00000023', '00000025']:
        #         cv2.imwrite(osp.join(save_dir, osp.basename(image_list[idx])[:-4] + area + '.png'), sal_map)
        #         # cv2.imwrite(osp.join(save_dir, osp.basename(image_list[idx])[:-4] + '_gtgray.png'), gt_map)
        #         # shutil.copy(image_list[idx], osp.join(save_dir, osp.basename(image_list[idx])[:-4] + '_gtcolor.png'))
        #    if i == 0:
        eval.add_batch(img_out[:, 0, :, :], label.unsqueeze(dim=0))

    F_beta_max, MAE = eval.get_metric()
    # print('Overall F_beta (Val): %.4f\t MAE (Val): %.4f' % (F_beta, MAE))
    return F_beta_max, MAE


def main(args, file_list):
    # read all the images in the folder
    image_list = list()
    label_list = list()
    with open(args.data_dir + '/' + file_list + '.txt') as fid:
        for line in fid:
            line_arr = line.split()
            image_list.append(args.data_dir + '/' + line_arr[0].strip())
            label_list.append(args.data_dir + '/' + line_arr[1].strip())

    # model = net.GAPNet(arch=args.arch, global_guidance=args.gbg, diverse_supervision=args.dds, attention=args.attention, kv_conc=args.kvc)
    model = net.GAPNet(arch=args.arch)
    if not osp.isfile(args.pretrained):
        print('Pre-trained model file does not exist...')
        exit(-1)

    state_dict = torch.load(args.pretrained, map_location='cpu')

    total_paramters = sum([np.prod(p.size()) for p in model.parameters()])
    # total_parameters_saved = sum(p.numel() for p in state_dict.values())
    print(f'Total network parameters: {total_paramters/1e6:.6f}M')
    # print('Total saved network parameters: ' + str(total_parameters_saved))

    model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    model.eval()
    # set to evaluation mode

    flops = FlopCountAnalysis(model, torch.rand(1, 3, 384, 384).to(device))
    print(f"total flops: {flops.total()/1e9:.4f}G")

    ######################################
    #### PyTorch Test [BatchSize 20] #####
    ######################################
    bs = 20
    x = torch.randn(bs, 3, 384, 384).to(device)
    for _ in range(50):
        # warm up
        y = model(x)
    from time import time
    total_t = 0
    for _ in range(100):
        start = time()
        y = model(x)
        # p = p + 1 # replace torch.cuda.synchronize()
        total_t += time() - start

    print("FPS", 100 / total_t * bs)
    print(f"PyTorch batchsize={bs} speed test completed, expected 450FPS for RTX 3090!")
    save_dir = osp.join(folder, file_list)
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    F_beta_max, MAE = test(args, model, image_list, label_list, save_dir)
    return F_beta_max, MAE


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--arch', default='vgg16', help='the backbone name of EDN, vgg16, resnet50, or mobilenetv2')
    parser.add_argument('--data_dir', default="./data-sod", help='Data directory')
    parser.add_argument('--width', type=int, default=384, help='Width of RGB image')
    parser.add_argument('--height', type=int, default=384, help='Height of RGB image')
    parser.add_argument('--savedir', default='./outputs', help='directory to save the results')
    parser.add_argument('--gpu', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU')
    parser.add_argument('--pretrained', default=None, help='Pretrained model')
    parser.add_argument('--add_dwconv', default=0, type=int)
    parser.add_argument('--last_channel', default=80, type=int)
    #parser.add_argument('--low_scale', default=8, type=int)
    parser.add_argument('--dds', default=1, type=int, help="diverse supervision")
    parser.add_argument('--gbg', default=1, type=int, help="global guidance")
    parser.add_argument('--igi', default=0, type=int, help="ignore index")
    parser.add_argument('--kvc', default=0, type=int, help="concatenate x1, x2 for k, v computation or not")
    parser.add_argument('--qc', default=1, type=int, help="concatenate x1, x2 for q computation or not")
    parser.add_argument('--attention', default="EA", choices=["EA", "SA"], type=str, help="attention mechanisms: self-attention, efficient-attention")
    parser.add_argument('--dilation_opt', default=1, choices=[1, 2], type=int, help="dilation option")
    parser.add_argument('--low_global_vit', default=0, type=int, help="use vit for edge/global feature fusion")
    parser.add_argument('--vit_dwconv', default=1, type=int, help="add dwconv in vit ffn")
    parser.add_argument('--supervision', default=1, choices=[0, 1, 2, 3, 4, 5], type=int, help="supervision signals")
    args = parser.parse_args()

    print('Called with args:')
    print(args)

    # data_lists1 = ["DUTS-TE", "DUT-OMRON", "HKU-IS", "ECSSD", "PASCAL-S"]
    data_lists1 = ["DUTS-TE", "DUT-OMRON", "HKU-IS", "ECSSD", "PASCAL-S", "SOC6K", "THUR15K"]
    # data_lists2 = ["THUR15K"]
    # data_lists = ["DUTS-TE"]
    folder = args.savedir
    device = torch.device('cuda') if args.gpu else torch.device('cpu')

    F_max_list, F_mean_list, MAE_list = [], [], []
    
    for data_index, data_list in enumerate(data_lists1):
        print("processing ", data_list)
        epoch_best = 30
        print(f"best epoch for {data_list} is: {epoch_best}")
        F_max, MAE = main(args, data_list)
        F_max_list.append(F_max)
        MAE_list.append(MAE)
        #F_mean_list.append(F_mean)
        print(F_max_list, MAE_list)
        #args.pretrained = folder
