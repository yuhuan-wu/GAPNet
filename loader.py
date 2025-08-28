import torch, os, sys, glob
import cv2
import numpy as np
from scipy import ndimage
from copy import deepcopy
import matlab
import matplotlib.pyplot as plt

class EvalLoader(torch.utils.data.Dataset):

    def __init__(self, gt_path, dt_path, eval_type='normal', postfix='.png'):
        self.gt_path = gt_path
        self.dt_path = dt_path
        self.imgs = [os.path.basename(x) for x in glob.glob(self.gt_path + '/*{}'.format(postfix))]
        self.nthresh = 99
        self.EPSILON = np.finfo(float.eps
        self.thresh = np.linspace(1. / (self.nthresh + 1), 1. - 1. / (self.nthresh + 1), self.nthresh)
        self.eval_type = eval_type

    def cal(self, gt, dt, idx):
        r = np.zeros(self.nthresh)
        p = np.zeros(self.nthresh)
        for t in range(self.nthresh):
            bi_res = dt > self.thresh[t]
            intersection = np.sum(np.logical_and(gt == bi_res, gt))
            r[t] = intersection * 1. / (np.sum(gt) + self.EPSILON)
            p[t] = intersection * 1. / (np.sum(bi_res) + self.EPSILON)
        mae = np.sum(np.fabs(gt - dt)) * 1. / (gt.shape[0] * gt.shape[1])

        return p, r, mae
    
    @staticmethod
    def imwrite(path, img):
        if not os.path.exists(path):
            cv2.imwrite(path.replace('data', 'data_mid'), img.astype(np.uint8)*255)

    def cal_body(self, gt, dt, idx):
        gt = gt.astype(np.uint8)
        gt_raw_body = cv2.distanceTransform(gt, distanceType=cv2.DIST_L2, maskSize=5)
        temp = gt_raw_body.flatten(); temp = temp[temp.nonzero()] # get nonzero flatten vector
        try:
            top30value = temp[np.argpartition(temp, kth=int(np.size(temp)*0.8))[int(np.size(temp)*0.8)]]
        except IndexError: # no foreground
            return np.ones(self.nthresh), np.ones(self.nthresh), 0

        gt_body = gt_raw_body > top30value
        self.imwrite(self.get_name(idx)[0][:-4]+'_center.png', gt_body.astype(np.uint8))
        #gt_body = gt_raw_body > gt_raw_body.mean() #+ gt_raw_body.std() # 2 sigma rule.
        gt_body_ignore = gt - gt_body
        r = np.zeros(self.nthresh)
        p = np.zeros(self.nthresh)
        for t in range(self.nthresh):
            bi_res = dt > self.thresh[t]
            intersection = np.sum(np.logical_and(gt == bi_res, gt) * gt_body)
            r[t] = intersection * 1. / (np.sum(gt_body) + self.EPSILON)
            p[t] = intersection * 1. / (np.sum(bi_res - bi_res * gt_body_ignore) + self.EPSILON)
        mae = np.sum(np.fabs(gt_body - dt) * gt_body) * 1. / (gt_body.sum()+1e-8)
        return p, r, mae


    def cal_detail(self, gt, dt, idx):
        gt = gt.astype(np.uint8)
        gt_raw_body = cv2.distanceTransform(gt, distanceType=cv2.DIST_L2, maskSize=5)
        # gt_body = gt_raw_body > gt_raw_body.mean() + gt_raw_body.std() * 1 # 2 sigma rule.
        #gt_raw_body_non_zero_mean = gt_raw_body.sum() / ((gt_raw_body>0).sum() + 1e-8)
        gt_detail = (gt_raw_body<5) * (gt_raw_body>0)
        self.imwrite(self.get_name(idx)[0][:-4]+'_edge.png', gt_detail.astype(np.uint8))
        #plt.imshow(gt_detail); plt.show()
        gt_detail_ignore = gt - gt_detail
        r = np.zeros(self.nthresh)
        p = np.zeros(self.nthresh)
        for t in range(self.nthresh):
            bi_res = dt > self.thresh[t]
            intersection = np.sum(np.logical_and(gt == bi_res, gt) * gt_detail)
            r[t] = intersection * 1. / (np.sum(gt_detail) + self.EPSILON)
            p[t] = intersection * 1. / (np.sum(bi_res - bi_res * gt_detail_ignore) + self.EPSILON)
        mae = np.sum(np.fabs(gt_detail - dt) * gt_detail) * 1. / (gt_detail.sum()+1e-8)
        return p, r, mae


    def cal_other(self, gt, dt, idx):
        gt = gt.astype(np.uint8)
        gt_raw_body = cv2.distanceTransform(gt, distanceType=cv2.DIST_L2, maskSize=5)
        temp = gt_raw_body.flatten(); temp = temp[temp.nonzero()] # get nonzero flatten vector
        try:
            top30value = temp[np.argpartition(temp, kth=int(np.size(temp)*0.8))[int(np.size(temp)*0.8)]]
        except IndexError: # no foreground
            return np.ones(self.nthresh), np.ones(self.nthresh), 0
        gt_body = gt_raw_body > top30value
        gt_detail = (gt_raw_body<5) * (gt_raw_body>0)
        gt_other = gt - np.logical_or(gt_body, gt_detail)
        self.imwrite(self.get_name(idx)[0][:-4]+'_other.png', gt_other.astype(np.uint8))
        #plt.imshow(gt_detail); plt.show()
        gt_other_ignore = gt - gt_other
        r = np.zeros(self.nthresh)
        p = np.zeros(self.nthresh)
        for t in range(self.nthresh):
            bi_res = dt > self.thresh[t]
            intersection = np.sum(np.logical_and(gt == bi_res, gt) * gt_other)
            r[t] = intersection * 1. / (np.sum(gt_other) + self.EPSILON)
            p[t] = intersection * 1. / (np.sum(bi_res - bi_res * gt_other_ignore) + self.EPSILON)
        mae = np.sum(np.fabs(gt_other - dt) * gt_other) * 1. / (gt_other.sum()+1e-8)
        return p, r, mae

 
        
    def get_name(self, idx):
        gt_name = os.path.join(self.gt_path, self.imgs[idx])
        dt_name = os.path.join(self.dt_path, self.imgs[idx])
        return gt_name, dt_name

    def __getitem__(self, idx):
        gt_name, dt_name = self.get_name(idx)
        gt = cv2.imread(gt_name, cv2.IMREAD_GRAYSCALE)
        flag = 0
        if os.path.exists(dt_name):
            dt = cv2.imread(dt_name, cv2.IMREAD_GRAYSCALE)
        else:
            dt = np.zeros(gt.shape, dtype=float
            flag = 1
            print("dt error!")
            print(dt_name)
        if dt.shape[0] != gt.shape[0] or dt.shape[1] != gt.shape[1]:
            dt = cv2.resize(dt, (gt.shape[1], gt.shape[0]), cv2.INTER_NEAREST)
        gt = (gt == 255).astype(float
        dt = dt.astype(float / 255
        if self.eval_type == 'body':
            p, r, mae = self.cal_body(gt, dt, idx)
        elif self.eval_type == 'detail':
            p, r, mae = self.cal_detail(gt, dt, idx)
        elif self.eval_type == 'normal':
            p, r, mae = self.cal(gt, dt, idx)
        elif self.eval_type == 'other':
            p, r, mae = self.cal_other(gt, dt, idx)
        return p, r, mae, flag

    def __len__(self):
        return len(self.imgs)
