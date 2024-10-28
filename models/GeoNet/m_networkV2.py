import torch.nn as nn
from models.GeoNet.m_blockV2 import FEB, Tail, Head, DownBlock, UpBlock, CubeEmbeding, CubeUnEmbeding, Norm, ACT
from einops import rearrange
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader
import torch
import jacksung.utils.fastnumpy as fnp


class GeoNet(nn.Module):
    def __init__(self, window_sizes, n_lgab, c_in, c_lgan, r_expand=4, down_sample=4, num_heads=8, task='pred',
                 downstage=2):
        super(GeoNet, self).__init__()
        self.window_sizes = window_sizes
        self.n_lgab = n_lgab
        self.c_in = c_in
        self.c_lgan = c_lgan
        self.r_expand = r_expand
        self.task = task
        self.down_sample = down_sample
        # define head module
        if self.task == 'prec':
            self.head = Head(self.c_lgan, self.c_in, self.down_sample)
        else:
            self.head = CubeEmbeding(self.c_lgan, self.c_in, self.down_sample)
            self.head_res = Head(self.c_lgan, self.c_in, self.down_sample)
        # self.head = Head(self.c_lgan, self.c_in * 2, self.down_sample)
        # define body module
        self.body = nn.ModuleList()
        self.downstage = downstage
        for i in range(self.n_lgab):
            if i / self.downstage in [1]:
                self.body.append(DownBlock(self.c_lgan, 2))
                self.c_lgan = self.c_lgan * 2
            elif (self.n_lgab - i) / self.downstage in [1]:
                self.body.append(UpBlock(self.c_lgan // 2, 2))
                self.c_lgan = self.c_lgan // 2
            self.body.append(
                FEB(self.c_lgan, self.r_expand, self.window_sizes[i % len(self.window_sizes)], num_heads=num_heads))
        # self.conv1 = nn.Conv2d(self.c_lgan, self.c_lgan, 3, 1, 1)
        # self.norm = Norm(self.c_lgan)
        # self.act = ACT()
        # self.conv2 = nn.Conv2d(self.c_lgan, self.c_lgan, 3, 1, 1)
        self.tail = Tail(self.c_lgan, 1 if self.task == 'prec' else self.c_in, down_sample)
        # self.tail = nn.ConvTranspose3d(self.c_lgan, 5 if self.task == 'prec' else self.c_in, kernel_size=(2, 2, 2),
        #                                stride=(2, 2, 2))
        # self.tail = CubeUnEmbeding(self.c_lgan, 5 if self.task == 'prec' else self.c_in, self.down_sample)

    def forward(self, f, x, roll=0):
        # head
        head_res = None
        if self.task == 'pred':
            head_res = self.head_res(x)
            x = torch.stack([x, f], dim=2)
        if roll > 0:
            x = torch.roll(x, shifts=roll, dims=-1)
        # f, x = rearrange(x, 'b c z h w->b (c z) h w'), rearrange(f, 'b c z h w->b (c z) h w')
        # x = nn.functional.interpolate(x, size=[1600, 2000], mode='bilinear')
        # f = nn.functional.interpolate(f, size=[1600, 2000], mode='bilinear')
        x = self.head(x)
        # shortcut = x
        # body
        x_res = None
        for idx, stage in enumerate(self.body):
            if idx / self.downstage in [1]:
                x_res = x
            x = stage(x)
            if (self.n_lgab + 1 - idx) / self.downstage in [1]:
                x += x_res
        if self.task == 'pred':
            x = head_res + x
        # tail
        x = self.tail(x)
        # x = x[:, :, 0, :, :]
        if roll > 0:
            x = torch.roll(x, shifts=-roll, dims=-1)
        # x = nn.functional.interpolate(x, size=[1607, 2008], mode='bilinear')
        return x

    def init_model(self):
        print('Initializing the model!')
        for m in self.children():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)

    def load(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            name = name[name.index('.') + 1:]
            if name in own_state.keys():
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                    # own_state[name].requires_grad = False
                except Exception as e:
                    err_log = f'While copying the parameter named {name}, ' \
                              f'whose dimensions in the model are {own_state[name].size()} and ' \
                              f'whose dimensions in the checkpoint are {param.size()}.'
                    if not strict:
                        print(err_log)
                    else:
                        raise Exception(err_log)
            elif strict:
                raise KeyError(f'unexpected key {name} in {own_state.keys()}')
            else:
                print(f'{name} not loaded by model')


if __name__ == '__main__':
    pass
