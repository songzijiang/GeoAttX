import shutil

import numpy as np
import yaml
from util import utils
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.datasetPred import Benchmark
from datetime import datetime
from jacksung.utils.time import RemainTime, Stopwatch, cur_timestamp_str
from jacksung.utils.data_convert import np2tif
from tqdm import tqdm
from util.norm_util import PredNormalization
from util.data_parallelV2 import BalancedDataParallel
from util.utils import getFY_coord, getFY_coord_min
from util.norm_util import PredNormalization, PrecNormalization
from dataset import datasetPred, datasetPrec

if __name__ == '__main__':
    device, args = utils.parse_config()
    step = args.current if args.current > 0 else args.p_step
    test_dataset = None
    if args.task == 'pred':
        test_dataset = datasetPred.Benchmark(args.pred_data_path, f'test{args.p_step}.txt', train=False,
                                             batch_size=args.batch_size)
    elif args.task == 'prec':
        test_dataset = datasetPrec.Benchmark(args.prec_data_path, f'test.txt', train=False, batch_size=args.batch_size)
    test_dataloader = DataLoader(dataset=test_dataset, num_workers=args.threads, batch_size=args.batch_size,
                                 shuffle=False, pin_memory=False, drop_last=False)
    model = utils.get_model(args)
    norm = None
    if args.task == 'pred':
        norm = PredNormalization(args.pred_data_path)
        norm.mean, norm.std = utils.data_to_device([norm.mean, norm.std], device, args.fp)
    elif args.task == 'prec':
        norm = PrecNormalization(args.prec_data_path)
        norm.mean_fy, norm.mean_era, norm.std_fy, norm.std_era = \
            utils.data_to_device([norm.mean_fy, norm.mean_era, norm.std_fy, norm.std_era], device, args.fp)
    # load pretrain
    if args.model_path is None:
        raise Exception('进行数据生成，请在config中指定 pretrain 参数')
    print('load pretrained model: {}!'.format(args.model_path))
    ckpt = torch.load(args.model_path)
    model.load(ckpt['model_state_dict'])
    model = model.to(device)
    if args.balanced_gpu0 >= 0:
        # balance multi gpus
        model = BalancedDataParallel(args.balanced_gpu0, model, device_ids=list(range(len(args.gpu_ids))))
    else:
        # multi gpus
        model = nn.DataParallel(model, device_ids=list(range(len(args.gpu_ids))))
    save_path = args.save_path
    timestamp = cur_timestamp_str()

    root_path = os.path.join(args.save_path, args.task + '-' + args.model + '-' + timestamp)
    os.makedirs(root_path, exist_ok=True)
    torch.set_grad_enabled(False)
    model = model.eval()
    rr = 0
    rmse = 0
    progress_bar = tqdm(total=len(test_dataset), desc='Infer')
    count = 0
    shutil.copyfile(args.model_path, f'{root_path}/model.pth')
    exp_params = vars(args)
    exp_params_name = os.path.join(root_path, 'config_saved.yml')
    with open(exp_params_name, 'w') as exp_params_file:
        yaml.dump(exp_params, exp_params_file, default_flow_style=False)
    for iter_idx, batch in enumerate(test_dataloader):
        f, n, p = utils.data_to_device(batch, device, args.fp)
        n_ = n.clone().cpu().numpy()
        if args.task == 'pred':
            f, n, p = norm.norm(f), norm.norm(n), norm.norm(p)
        elif args.task == 'prec':
            f, n = norm.norm(f, norm_type='fy'), norm.norm(n, norm_type='fy')
        # roll = random.randint(0, now_t.shape[-1] - 1)
        roll = 0
        y_ = None
        if args.task == 'pred':
            y_ = model(f, n, roll)
            # y_ = model(n, n, roll)
        elif args.task == 'prec':
            for cur in range(4):
                if y_ is None:
                    y_ = model(f[:, cur], n[:, cur], roll)
                else:
                    y_ += model(f[:, cur], n[:, cur], roll)
        b, c, h, w = y_.shape
        if args.task == 'pred':
            y_ = norm.denorm(y_)
            p = norm.denorm(p)
        elif args.task == 'prec':
            y_ = norm.denorm(y_, norm_type='era')
            p = norm.denorm(p, norm_type='era')
        y_ = y_.cpu().numpy()
        p = p.cpu().numpy()
        coord = None
        values = None
        FY_values = [{'value': [str(x) for x in list(range(7, 16))]}]
        ERA5_values = [{'value': ['u10', 'v10', 't2m', 'msl', 'tp']}]
        if args.task == 'pred':
            coord = getFY_coord_min(133)
            values = FY_values
        elif args.task == 'prec':
            coord = getFY_coord_min(133)
            values = ERA5_values
        for idx, each_y in enumerate(y_):
            each_p = p[idx]
            each_n = n_[idx]
            data_idx = str(iter_idx * args.batch_size + idx)
            img_path = os.path.join(root_path, data_idx)
            os.makedirs(img_path, exist_ok=True)
            np2tif(each_p, img_path, f'{data_idx}_target', coord=coord, dim_value=values, print_log=False)
            np2tif(each_y, img_path, f'{data_idx}_y_', coord=coord, dim_value=values, print_log=False)
            if args.task == 'pred':
                np2tif(each_n, img_path, f'{data_idx}_n', coord=coord, dim_value=values, print_log=False)
            elif args.task == 'prec':
                for i in range(4):
                    np2tif(each_n[i], img_path, f'{data_idx}_n_{i}', coord=coord, dim_value=FY_values, print_log=False)
        progress_bar.update(len(f))
    progress_bar.close()
