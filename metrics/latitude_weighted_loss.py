import torch
from torch import nn
import numpy as np
from einops import rearrange
import sys
from jacksung.utils.data_convert import np2tif

sys.path.append('../../..')
from util.utils import min_x_range, x_range, getFY_coord


class LatLonLoss(nn.Module):

    def __init__(self):
        super(LatLonLoss, self).__init__()
        self.weights = None

    def set_task_type(self, task_type='pred'):
        ran = None
        if task_type == 'pred':
            ran = x_range
        elif task_type == 'prec':
            ran = min_x_range
        step = (ran['top'] - ran['bottom']) / ran['height']
        weights_lat = np.arange(ran['bottom'], ran['top'], step)
        weights_lat = np.cos(weights_lat * np.pi / 180)
        weights_lat = ran['height'] * weights_lat / np.sum(weights_lat)
        weights_lat = np.tile(weights_lat, (ran['height'], 1))

        step = (ran['right'] - ran['left']) / ran['width']
        weights_lon = np.arange(ran['left'], ran['right'], step)
        weights_lon = np.cos(weights_lon * np.pi / 180)
        weights_lon = ran['width'] * weights_lon / np.sum(weights_lon)
        weights_lon = np.tile(weights_lon, (ran['width'], 1))

        self.weights = torch.from_numpy(weights_lat * weights_lon.T)
        # self.weights = torch.from_numpy(x_range['height'] * self.weights_lat / np.sum(self.weights_lat))

    def forward(self, output, target):
        if self.weights.device != output.device:
            self.weights = self.weights.to(output.device)
        loss = torch.abs(output - target) * self.weights
        loss = torch.mean(loss)
        return loss


class LatitudeLoss(nn.Module):

    def __init__(self):
        super(LatitudeLoss, self).__init__()
        self.weights_lat = None

    def set_task_type(self, type_task='pred'):
        ran = min_x_range
        step = (ran['top'] - ran['bottom']) / ran['height']
        self.weights_lat = np.arange(ran['bottom'], ran['top'], step)
        self.weights_lat = np.cos(self.weights_lat * np.pi / 180)
        self.weights_lat = torch.from_numpy(ran['height'] * self.weights_lat / np.sum(self.weights_lat))

    def forward(self, output, target):
        if self.weights_lat.device != output.device:
            self.weights_lat = self.weights_lat.to(output.device)
        loss = rearrange(torch.abs(output - target), 'b c h w->b c w h') * self.weights_lat
        loss = torch.mean(loss)
        return loss


if __name__ == '__main__':
    loss = LatitudeLoss()
    np2tif(loss.weights.numpy(), out_name='loss_weight', coord=getFY_coord(133))
