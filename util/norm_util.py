import os.path

import torch
import numpy as np
from einops import rearrange


class PredNormalization:
    def __init__(self, data_path):
        self.mean = np.load(os.path.join(data_path, 'mean_level.npy')).astype(np.float32)[2:]
        self.std = np.load(os.path.join(data_path, 'std_level.npy')).astype(np.float32)[2:]
        self.mean = torch.from_numpy(self.mean)
        self.std = torch.from_numpy(self.std)

    def norm(self, data):
        data = rearrange(data, 'b c h w->b h w c')
        data = (data - self.mean) / self.std
        return rearrange(data, 'b h w c->b c h w')

    def denorm(self, data):
        data = rearrange(data, 'b c h w->b h w c')
        data = data * self.std + self.mean
        return rearrange(data, 'b h w c->b c h w')


class PrecNormalization:
    def __init__(self, data_path):
        self.mean_qpe = torch.from_numpy(
            np.load(os.path.join(data_path, 'mean_level_qpe.npy')).astype(np.float32))
        self.std_qpe = torch.from_numpy(
            np.load(os.path.join(data_path, 'std_level_qpe.npy')).astype(np.float32))
        self.mean_fy = torch.from_numpy(
            np.load(os.path.join(data_path, 'mean_level_fy.npy')).astype(np.float32).mean(axis=0)[2:])
        self.std_fy = torch.from_numpy(
            np.load(os.path.join(data_path, 'std_level_fy.npy')).astype(np.float32).mean(axis=0)[2:])

    def norm(self, data, norm_type='fy'):
        if norm_type == 'fy':
            data = rearrange(data, 'b t c h w->b h w t c')
            data = (data - self.mean_fy) / self.std_fy
            return rearrange(data, 'b h w t c->b t c h w')
        elif norm_type == 'qpe':
            data = rearrange(data, 'b c h w->b h w c')
            data = (data - self.mean_qpe) / self.std_qpe
            return rearrange(data, 'b h w c->b c h w')

    def denorm(self, data, norm_type='fy'):
        if norm_type == 'fy':
            data = rearrange(data, 'b t c h w->b h w t c')
            data = data * self.std_fy + self.mean_fy
            return rearrange(data, 'b h w t c->b t c h w')
        elif norm_type == 'qpe':
            data = rearrange(data, 'b c h w->b h w c')
            data = data * self.std_qpe + self.mean_qpe
            return rearrange(data, 'b h w c->b c h w')
