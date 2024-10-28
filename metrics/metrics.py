import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from pytorch_msssim import ssim
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics import R2Score
from torchmetrics.regression import MeanSquaredError
import importlib
from einops import rearrange
import cv2
from datetime import datetime


def compute_rmse(da_fc, da_true):
    error = da_fc - da_true
    error = error ** 2
    number = torch.sqrt(error.mean((-2, -1)))
    return number.mean()


class Metrics:
    def __init__(self):
        self.psnr = PeakSignalNoiseRatio()
        self.ssim = StructuralSimilarityIndexMeasure()
        self.rr = R2Score()

    def calc_psnr(self, sr, hr):
        return self.psnr(sr, hr)

    def calc_ssim(self, sr, hr):
        return self.ssim(sr, hr)

    def calc_rmse(self, sr, hr):
        return compute_rmse(sr, hr)

    def calc_rr(self, sr, hr):
        return self.rr(sr.flatten(), hr.flatten())

    def print_metrics(self, sr, hr, print_log=True):
        rr = float(self.calc_rr(sr, hr))
        rmse = float(self.calc_rmse(sr, hr))
        ssim = float(self.calc_ssim(sr, hr))
        psnr = float(self.calc_psnr(sr, hr))
        if print_log:
            print(rf'rr: {rr} rmse: {rmse} ssim: {ssim} psnr: {psnr}')
        return {'rr': rr, 'rmse': rmse, 'ssim': ssim, 'psnr': psnr}


def img2tensor(img):
    if type(img) == str:
        img = cv2.imread(img, -1)
    img = torch.from_numpy(img)
    img = rearrange(img, ' (b c h) w->b c h w', b=1, c=1)
    return img


if __name__ == '__main__':
    # preds = torch.rand(2, 3, 3, 5)
    # target = torch.rand(2, 3, 3, 5)
    # m = Metrics()
    # print(m.calc_rr(preds, target))
    m = Metrics()
    img1 = img2tensor(r'C:\Users\ECNU\Desktop\fyp\target_20231220_053000-13.tif')
    img2 = img2tensor(r'C:\Users\ECNU\Desktop\fyp\20231220_053000-13-4.tif')
    m.print_metrics(img2, img1)
