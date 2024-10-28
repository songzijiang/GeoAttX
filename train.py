import torch
import math

import yaml
from jacksung.utils.data_convert import np2tif

from util import utils
import os
import sys
import random

import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from dataset import datasetPred, datasetPrec
import numpy as np

from metrics.latitude_weighted_loss import LatitudeLoss, LatLonLoss
from tqdm import tqdm
from jacksung.utils.log import LogClass, oprint
from jacksung.utils.time import RemainTime, Stopwatch, cur_timestamp_str
from datetime import datetime
import jacksung.utils.fastnumpy as fnp
from jacksung.utils.log import StdLog
from util.data_parallelV2 import BalancedDataParallel
from util.norm_util import PredNormalization, PrecNormalization
from metrics.metrics import Metrics
from einops import rearrange

from util.utils import getFY_coord_min

# os.environ["NCCL_DEBUG"] = "INFO"
# os.environ['TORCH_DISTRIBUTED_DEBUG']= "INFO"
# os.environ['TORCH_SHOW_CPP_STACKTRACES']= "1"

if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)

    device, args = utils.parse_config()
    # definitions of model
    model = utils.get_model(args)

    # load pretrain
    if args.pretrain is not None:
        print('load pretrained model: {}!'.format(args.pretrain))
        ckpt = torch.load(args.pretrain)
        model.load(ckpt['model_state_dict'], strict=False)
    # definition of loss and optimizer
    loss_func = eval(args.loss)
    if args.loss == 'LatitudeLoss()':
        loss_func.set_task_type(args.task)
    if args.fp == 16:
        eps = 1e-3
    elif args.fp == 64:
        eps = 1e-13
    else:
        eps = 1e-8
    optimizer = eval(f'torch.optim.{args.optimizer}(model.parameters(), lr=args.lr, eps=eps)')
    scheduler = MultiStepLR(optimizer, milestones=args.decays, gamma=args.gamma)

    # resume training
    if args.resume is not None:
        ckpt_files = os.path.join(args.resume, 'models', "model_latest.pt")
        if len(ckpt_files) != 0:
            ckpt = torch.load(ckpt_files)
            prev_epoch = ckpt['epoch']
            start_epoch = prev_epoch + 1
            model.load(ckpt['model_state_dict'], strict=True)
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
            stat_dict = ckpt['stat_dict']
            # reset folder and param
            experiment_path = args.resume
            experiment_model_path = os.path.join(experiment_path, 'models')
            print('Select {} file, resume training from epoch {}.'.format(ckpt_files, start_epoch))
        else:
            raise Exception(f'{os.path.join(args.resume, "models", "model_latest.pt")}中无有效的ckpt_files')
    else:
        start_epoch = 1
        # auto-generate the output log name
        experiment_name = None
        timestamp = cur_timestamp_str()
        experiment_name = '{}-{}-{}'.format(args.task + ('' if args.task == 'prec' else 'x' + str(args.p_step)),
                                            args.model if args.log_name is None else args.log_name,
                                            timestamp)
        experiment_path = os.path.join(args.log_path, experiment_name)
        stat_dict = utils.get_stat_dict(
            (
                ('val-loss', float('inf'), '<'),
                ('RMSE', float('inf'), '<'),
                ('RR', float('0'), '>')
            )
        )
        # create folder for ckpt and stat
        if not os.path.exists(experiment_path):
            os.makedirs(experiment_path)
        experiment_model_path = os.path.join(experiment_path, 'models')
        if not os.path.exists(experiment_model_path):
            os.makedirs(experiment_model_path)
        # save training parameters
        exp_params = vars(args)
        exp_params_name = os.path.join(experiment_path, 'config_saved.yml')
        with open(exp_params_name, 'w') as exp_params_file:
            yaml.dump(exp_params, exp_params_file, default_flow_style=False)
        model.init_model()
    print('Experiment path: {}'.format(experiment_path))
    model = model.to(device)
    if args.fp == 16:
        model = model.half()
    elif args.fp == 64:
        model = model.double()

    if args.balanced_gpu0 >= 0:
        # balance multi gpus
        model = BalancedDataParallel(args.balanced_gpu0, model, device_ids=list(range(len(args.gpu_ids))))
    else:
        # multi gpus
        model = nn.DataParallel(model, device_ids=list(range(len(args.gpu_ids))))
    log_name = os.path.join(experiment_path, 'log.txt')
    warning_path = os.path.join(experiment_path, 'warning.txt')
    stat_dict_name = os.path.join(experiment_path, 'stat_dict.yml')
    sys.stdout = StdLog(filename=log_name, common_path=warning_path)

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    if args.resume is None:
        print('Total Number of Parameters:' + str(round(num_params / 1024 ** 2, 2)) + 'M')
        print('Data path: ' + args.pred_data_path if args.task == 'pred' else args.prec_data_path)
        print(f'Now {args.task} task selected.')
    train_dataset, valid_dataset = None, None
    if args.task == 'pred':
        train_dataset = datasetPred.Benchmark(args.pred_data_path, f'train{args.p_step}.txt', train=True,
                                              repeat=args.repeat, batch_size=args.batch_size)
        valid_dataset = datasetPred.Benchmark(args.pred_data_path, f'test{args.p_step}.txt', train=False,
                                              batch_size=args.batch_size, split_size=args.split_size)
    elif args.task == 'prec':
        train_dataset = datasetPrec.Benchmark(args.prec_data_path, f'train.txt', train=True, repeat=args.repeat,
                                              batch_size=args.batch_size)
        valid_dataset = datasetPrec.Benchmark(args.prec_data_path, f'test.txt', train=False, batch_size=args.batch_size,
                                              split_size=args.split_size)
    # create dataset for training and validating
    train_dataloader = DataLoader(dataset=train_dataset, num_workers=args.threads, batch_size=args.batch_size,
                                  shuffle=False, pin_memory=False, drop_last=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, num_workers=args.threads, batch_size=args.batch_size,
                                  shuffle=False, pin_memory=False, drop_last=False)
    # start training

    rt = RemainTime(args.epochs)
    rt.update(update_step=start_epoch - 1, print_log=False)
    cloudLogName = experiment_path.split(os.sep)[-1]
    log = LogClass(args.cloudlog == 'on')
    log.send_log('Start training', cloudLogName)
    log_every = max(len(train_dataloader) // args.log_lines, 1)
    norm = None
    if args.task == 'pred':
        norm = PredNormalization(args.pred_data_path)
        norm.mean, norm.std = utils.data_to_device([norm.mean, norm.std], device, args.fp)
    elif args.task == 'prec':
        norm = PrecNormalization(args.prec_data_path)
        norm.mean_fy, norm.mean_qpe, norm.std_fy, norm.std_qpe = \
            utils.data_to_device([norm.mean_fy, norm.mean_qpe, norm.std_fy, norm.std_qpe], device, args.fp)

    m = Metrics()
    m.psnr, m.ssim, m.rr = utils.data_to_device([m.psnr, m.ssim, m.rr], device, args.fp)
    sw = Stopwatch()
    for epoch in range(start_epoch, args.epochs + 1):
        sw.reset()
        epoch_loss = 0.0
        count = 0
        stat_dict['epochs'] = epoch
        model = model.train()
        opt_lr = scheduler.get_last_lr()
        print()
        print('##===============-fp{}- Epoch: {}, lr: {} =================##'.format(args.fp, epoch, opt_lr))
        train_dataloader.check_worker_number_rationality()
        # training the model
        for iter_idx, batch in enumerate(train_dataloader):
            bs = len(batch[0])
            optimizer.zero_grad()
            roll = 0
            # roll = random.randint(0, now_t.shape[-1] - 1)
            f, n, p, y_ = None, None, None, None
            if args.task == 'pred':
                f, n, p = utils.data_to_device(batch, device, args.fp)
                f, n, p = norm.norm(f), norm.norm(n), norm.norm(p)
                y_ = model(f, n, roll)
            elif args.task == 'prec':
                n, p = utils.data_to_device(batch, device, args.fp)
                n, p = norm.norm(n, norm_type='fy'), norm.norm(p, norm_type='qpe')
                for cur in range(4):
                    if y_ is None:
                        y_ = model(n[:, cur], n[:, cur], roll)
                    else:
                        y_ += model(n[:, cur], n[:, cur], roll)
            loss = loss_func(y_, p)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss) * bs
            count += bs
            # print log
            if (iter_idx + 1) % log_every == 0:
                cur_steps = (iter_idx + 1) * args.batch_size
                total_steps = len(train_dataloader) * args.batch_size
                fill_width = math.ceil(math.log10(total_steps))
                cur_steps = str(cur_steps).zfill(fill_width)
                epoch_width = math.ceil(math.log10(args.epochs))
                cur_epoch = str(epoch).zfill(epoch_width)
                avg_loss = epoch_loss / count
                epoch_loss = 0
                count = 0
                stat_dict['losses'].append(avg_loss)
                oprint('Epoch:{}, {}/{}, Loss: {:.4f}, T:{}'.format(
                    cur_epoch, cur_steps, total_steps, avg_loss, sw.reset()))
        # validating the model
        if epoch % args.test_every == 0:
            torch.set_grad_enabled(False)
            model = model.eval()
            epoch_loss = 0
            rr = 0
            rmse = 0
            progress_bar = tqdm(total=len(valid_dataset), desc='Infer')
            count = 0
            # compare = [[0, 0] for i in range(9)]
            for iter_idx, batch in enumerate(valid_dataloader):
                bs = len(batch[0])
                optimizer.zero_grad()
                roll = 0
                # roll = random.randint(0, now_t.shape[-1] - 1)
                y_, f, n, p = None, None, None, None
                if args.task == 'pred':
                    f, n, p = utils.data_to_device(batch, device, args.fp)
                    f, n, p = norm.norm(f), norm.norm(n), norm.norm(p)
                    y_ = model(f, n, roll)
                elif args.task == 'prec':
                    n, p = utils.data_to_device(batch, device, args.fp)
                    n, p = norm.norm(n, norm_type='fy'), norm.norm(p, norm_type='qpe')
                    for cur in range(4):
                        if y_ is None:
                            y_ = model(n[:, cur], n[:, cur], roll)
                        else:
                            y_ += model(n[:, cur], n[:, cur], roll)
                loss = loss_func(y_, p)
                if args.task == 'pred':
                    y_ = norm.denorm(y_)
                    p = norm.denorm(p)
                elif args.task == 'prec':
                    y_ = norm.denorm(y_, norm_type='qpe')
                    p = norm.denorm(p, norm_type='qpe')
                m_idx = -1
                y_ = rearrange(y_[:, m_idx, :, :], '(b c) h w->b c h w', c=1)
                p = rearrange(p[:, m_idx, :, :], '(b c) h w->b c h w', c=1)
                rr += float(m.calc_rr(y_, p)) * bs
                rmse += float(m.calc_rmse(y_, p)) * bs
                epoch_loss += float(loss) * bs
                count += bs
                progress_bar.update(bs)
            # print(compare)
            progress_bar.close()
            epoch_loss = epoch_loss / count
            rr = rr / count
            rmse = rmse / count
            log_out = utils.make_best_metric(stat_dict,
                                             (
                                                 ('val-loss', float(epoch_loss)), ('RMSE', rmse), ('RR', rr)
                                             ),
                                             epoch, (experiment_model_path, model, optimizer, scheduler),
                                             (log, args.epochs, cloudLogName))
            # print log & flush out
            print(log_out)
            # save stat dict
            # save training parameters
            with open(stat_dict_name, 'w') as stat_dict_file:
                yaml.dump(stat_dict, stat_dict_file, default_flow_style=False)
            torch.set_grad_enabled(True)
            model = model.train()
        # update scheduler
        scheduler.step()
        rt.update()
    log.send_log('Training Finished!', cloudLogName)
    utils.draw_lines(stat_dict_name)
    print('Saved path: {}'.format(experiment_path))
