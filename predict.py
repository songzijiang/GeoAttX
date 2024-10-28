import os.path
from datetime import datetime, timedelta

from scipy.ndimage import zoom

from util import utils
import torch
from jacksung.utils.time import RemainTime, Stopwatch, cur_timestamp_str
from util.norm_util import PredNormalization, PrecNormalization
import numpy as np
from jacksung.utils.data_convert import np2tif
from util.utils import getFY_coord, prase_filename, get_reference, getNPfromHDF, getFY_coord_min, getFY_coord_clip, \
    getNPfromHDFClip
from einops import rearrange
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.datasetPred import Benchmark
from util.data_parallelV2 import BalancedDataParallel
import netCDF4 as nc
import cv2


class NoFileException(Exception):
    def __init__(self, file_name):
        self.file_name = file_name
        super().__init__(f'No such file: {file_name}')


class GeoAttX_I:
    def __init__(self, data_path, x1_path, x4_path, x12_path, root_path=None, config='predict.yml'):
        self.f, self.n, self.ys = None, None, None
        self.data_path = data_path
        self.device, self.args = utils.parse_config(config)
        self.x1 = self.load_model(x1_path)
        self.x4 = self.load_model(x4_path)
        self.x12 = self.load_model(x12_path)
        # self.x48 = self.load_model(x48_path)
        self.timestamp = cur_timestamp_str()
        self.root_path = os.path.join(root_path if root_path else self.args.save_path,
                                      'pred' + '-' + self.args.model + '-' + self.timestamp)
        self.norm = PredNormalization(self.args.pred_data_path)
        self.norm.mean, self.norm.std = utils.data_to_device([self.norm.mean, self.norm.std], self.device, self.args.fp)
        self.ld = None

    def get_root_path(self):
        return self.root_path

    def load_model(self, path):
        model = utils.get_model(self.args, task_type='pred')
        ckpt = torch.load(path, map_location=torch.device(self.device))
        model.load(ckpt['model_state_dict'])
        model = model.to(self.device)
        if self.args.balanced_gpu0 >= 0:
            # balance multi gpus
            model = BalancedDataParallel(self.args.balanced_gpu0, model, device_ids=list(range(len(self.args.gpu_ids))))
        else:
            # multi gpus
            model = nn.DataParallel(model, device_ids=list(range(len(self.args.gpu_ids))))
        model = model.eval()
        model.requires_grad_(False)
        return model

    def save(self, file_name, ys):
        file_info = prase_filename(file_name)
        ld = int(file_info["position"])
        for idx, (k, y) in enumerate(ys.items()):
            # coord = getFY_coord_min(ld)
            coord = getFY_coord_clip()
            np2tif(y, save_path=self.root_path, out_name=f'{k.strftime("%Y%m%d_%H%M%S")}', coord=coord,
                   dtype=np.float32, print_log=False, dim_value=[{'value': [str(x) for x in list(range(9, 16))]}])
            td = k - file_info['start']
            mins = td.days * 24 * 60 + td.seconds // 60
            target_filename = self.get_filename(file_name, mins)
            p_path = self.get_path_by_filename(target_filename)
            if idx >= 1 and os.path.exists(p_path):
                coord = getFY_coord_clip()
                p_data = getNPfromHDFClip(self.ld, p_path)[2:, :, :]
                np2tif(p_data, save_path=self.root_path, out_name=f'target_{k.strftime("%Y%m%d_%H%M%S")}',
                       print_log=False, coord=coord, dtype=np.float32,
                       dim_value=[{'value': [str(x) for x in list(range(9, 16))]}])
        print(f'data saved in {self.root_path}')
        with open(os.path.join(self.root_path, 'info.log'), 'w') as f:
            f.write(f'输入数据：{file_info["start"]} {file_info["position"]} {file_info["end"]}\n')
            for k, y in ys.items():
                f.write(f'预测：{k}\n')

    def get_filename(self, file_name, mins):
        file_info = prase_filename(file_name)
        new_file_name = file_name.replace(
            f'{file_info["start"].strftime("%Y%m%d%H%M%S")}_{file_info["end"].strftime("%Y%m%d%H%M%S")}',
            f'{(file_info["start"] + timedelta(minutes=mins)).strftime("%Y%m%d%H%M%S")}_'
            f'{(file_info["end"] + timedelta(minutes=mins)).strftime("%Y%m%d%H%M%S")}')
        return new_file_name

    def get_path_by_filename(self, file_name):
        file_info = prase_filename(file_name)
        return f'{self.data_path}/{file_info["start"].year}/{file_info["start"].month}/{file_info["start"].day}/{file_name}'

    def numpy2tensor(self, f_data):
        f_data = torch.from_numpy(f_data)
        f = utils.data_to_device([f_data], self.device, self.args.fp)[0]
        f = rearrange(f, '(b c) h w -> b c h w', b=1)
        return f

    def get_exist_by_filename_and_mins(self, file_name, mins):
        f_file_name = self.get_filename(file_name, mins)
        f_path = self.get_path_by_filename(f_file_name)
        if not os.path.exists(f_path):
            raise NoFileException(f_path)
        f_data = getNPfromHDFClip(self.ld, f_path)[2:, :, :]
        # f_data = zoom(f_data.astype(np.float32), (1, 1 / 5, 1 / 5))
        return self.numpy2tensor(f_data)

    def mean_std2Tensor(self, in_data, h, w):
        in_data = rearrange(in_data, '(b h w) c->b c h w', h=1, w=1)
        in_data = in_data.expand(1, 7, h, w)
        return in_data

    def secdOrderStd(self, t_data, mean0, std0, mean, std):
        h, w = t_data.shape[2], t_data.shape[3]
        t_data = (t_data - self.mean_std2Tensor(mean0, h, w)) / self.mean_std2Tensor(std0, h, w)
        t_data = t_data * self.mean_std2Tensor(std, h, w) + self.mean_std2Tensor(mean, h, w)
        return t_data

    def predict(self, file_name, step=360, p_steps=(48, 12, 4, 1)):
        try:
            file_info = prase_filename(file_name)
            self.ld = int(file_info["position"])
            step = step // 15
            # if step > 24:
            #     step = 24
            print(f'当前时刻:{file_info["start"]}\n预测长度:{step * 15}分钟')
            task_progress = []
            while step > 0:
                for p_step in p_steps:
                    if step >= p_step:
                        task_progress.append(p_step)
                        step -= p_step
                        break
            task_progress.reverse()
            print(f'正在预测:{file_info["start"] + timedelta(minutes=sum(task_progress) * 15)}...')
            n = self.get_exist_by_filename_and_mins(file_name, 0)
            now_date = file_info["start"]
            porcess_list = {now_date: n.detach().cpu().numpy()[0]}
            n = self.norm.norm(n)
            o_mean = torch.mean(n, dim=(2, 3))
            o_std = torch.std(n, dim=(2, 3))
            for step in task_progress:
                pre_date = now_date - timedelta(minutes=15 * step)
                now_date += timedelta(minutes=15 * step)
                print(f'predicting {now_date}...')
                if pre_date in porcess_list:
                    f = porcess_list[pre_date]
                    f = self.numpy2tensor(f)
                else:
                    f = self.get_exist_by_filename_and_mins(file_name,
                                                            -int((file_info["start"] - pre_date).seconds / 60))
                f = self.norm.norm(f)
                y_ = eval(f'self.x{step}(f, n)')
                # mean and std再标准化
                y_mean = torch.mean(y_, dim=(2, 3))
                y_std = torch.std(y_, dim=(2, 3))
                # 二次标准化
                y_ = self.secdOrderStd(y_, y_mean, y_std, o_mean, o_std)
                # y_ = torch.Tensor(np.zeros((1, 7, 2400, 2400), dtype=np.float32)).to(self.device)
                # for i in range(5):
                #     for j in range(5):
                #         n_ = n[:, :, 480 * i:480 * (i + 1), 480 * j:480 * (j + 1)]
                #         f_ = f[:, :, 480 * i:480 * (i + 1), 480 * j:480 * (j + 1)]
                #         y_[:, :, 480 * i:480 * (i + 1), 480 * j:480 * (j + 1)] = eval(f'self.x{step}(f_, n_)')
                del f
                n = y_
                porcess_list[now_date] = self.norm.denorm(y_).detach().cpu().numpy()[0]
        except NoFileException as e:
            os.makedirs(self.root_path, exist_ok=True)
            with open(os.path.join(self.root_path, 'err.log'), 'a') as f:
                filename = e.file_name.split(os.sep)[-1]
                file_info = prase_filename(filename)
                f.write(f'Not exist {file_info["start"]}\n')
            return {}
        return porcess_list


class GeoAttX_P:
    def __init__(self, model_path, root_path=None, config='predict_qpe.yml'):
        self.device, self.args = utils.parse_config(config)
        self.model = self.load_model(model_path)
        self.timestamp = cur_timestamp_str()
        self.root_path = os.path.join(root_path if root_path else self.args.save_path,
                                      'prec' + '-' + self.args.model + '-' + self.timestamp)

    def get_root_path(self):
        return self.root_path

    def load_model(self, path):
        model = utils.get_model(self.args, task_type='prec')
        ckpt = torch.load(path, map_location=torch.device(self.device))
        model.load(ckpt['model_state_dict'])
        model = model.to(self.device)
        if self.args.balanced_gpu0 >= 0:
            # balance multi gpus
            model = BalancedDataParallel(self.args.balanced_gpu0, model, device_ids=list(range(len(self.args.gpu_ids))))
        else:
            # multi gpus
            model = nn.DataParallel(model, device_ids=list(range(len(self.args.gpu_ids))))
        model = model.eval()
        model.requires_grad_(False)
        return model

    def save(self, y, save_name, info_log=True):
        np2tif(y, save_path=self.root_path, out_name=save_name, coord=getFY_coord_clip(), dtype=np.float32,
               print_log=False, dim_value=[{'value': ['qpe']}])
        print(f'data saved in {self.root_path}')
        if info_log:
            with open(os.path.join(self.root_path, 'info.log'), 'w') as f:
                f.write(f'反演：{save_name}\n')

    def predict(self, fy_npy_path):
        try:
            print(f'正在反演:{fy_npy_path}...')
            if not os.path.exists(fy_npy_path):
                raise NoFileException(fy_npy_path)
            n_data = np.load(fy_npy_path)
            n_data = torch.from_numpy(n_data)
            norm = PrecNormalization(self.args.prec_data_path)
            norm.mean_fy, norm.mean_qpe, norm.std_fy, norm.std_qpe = \
                utils.data_to_device([norm.mean_fy, norm.mean_qpe, norm.std_fy, norm.std_qpe], self.device,
                                     self.args.fp)
            n_data = utils.data_to_device([n_data], self.device, self.args.fp)[0]
            n_data = rearrange(n_data, '(b t c) h w -> b t c h w', b=1, t=1)
            n = norm.norm(n_data, norm_type='fy')[:, 0, :, :, :]
            y_ = self.model(n, n)
            y = norm.denorm(y_, norm_type='qpe').detach().cpu().numpy()[0]
            return y
        except NoFileException as e:
            os.makedirs(self.root_path, exist_ok=True)
            with open(os.path.join(self.root_path, 'err.log'), 'a') as f:
                filename = e.file_name.split(os.sep)[-1]
                file_info = prase_filename(filename)
                f.write(f'Not exist {file_info["start"]}\n')
            return None


if __name__ == '__main__':
    x1_path = '/home/ubuntu/FYpredict/experiments/predx1-geonet-2024-0903-1724-3260/models/model_40.pt'
    x4_path = '/home/ubuntu/FYpredict/experiments/predx4-geonet-2024-0718-2348-1154/models/model_latest.pt'
    x12_path = '/home/ubuntu/FYpredict/experiments/predx12-geonet-2024-0918-1440-2990/models/model_11.pt'
    default_file = 'FY4B-_AGRI--_N_DISK_1330E_L1-_FDI-_MULT_NOM_20231204000000_20231204001459_4000M_V0001.HDF'
    # default_file = 'FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240601000000_20240601001459_4000M_V0001.HDF'
    default_minutes = 60
    x1_path = input(f'please input the X1 pretrained model: ({x1_path})') or x1_path
    x4_path = input(f'please input the X4 pretrained model: ({x4_path})') or x4_path
    x12_path = input(f'please input the X12 pretrained model: ({x12_path})') or x12_path
    file_name = input(f'please input the filename of current data: ({default_file})') or default_file
    minutes = int(input(f'please input the minutes to predict: ({default_minutes})') or default_minutes)
    # data_path = '/mnt/data1/szj/FY/downloaded_file'
    data_path = '/mnt/data1/szj/FY/recent_data'
    for i in range(0, 24):
        for j in range(0, 4):
            file_name = f'FY4B-_AGRI--_N_DISK_1050E_L1-_FDI-_MULT_NOM_20240808{i:02}{j * 15:02}00_20240808{i:02}{(j + 1) * 15 - 1:02}59_4000M_V0001.HDF'
            file_path = os.path.join(data_path, '2024', '8', '8', file_name)
            if os.path.exists(file_path):
                I_net = GeoAttX_I(data_path, x1_path, x4_path, x12_path,
                                  root_path=f'/mnt/data1/szj/FY/240808/{i:02}{j * 15:02}')
                ys = I_net.predict(file_name, minutes, p_steps=(1,))
                I_net.save(file_name, ys)
                # *******************************************************************************************************
                model_path = '/home/ubuntu/FYpredict/experiments/prec-geonet-2024-0724-1803-5860/models/model_latest.pt'
                net = GeoAttX_P(model_path, root_path=f'/mnt/data1/szj/FY/recent_data/2024/{i:02}{j * 15:02}')
                input_data_path = os.path.join(net.get_root_path(), 'input.npy')
                file_info = prase_filename(file_name)
                os.makedirs(net.get_root_path(), exist_ok=True)
                np.save(input_data_path, getNPfromHDFClip(133, file_path)[2:, , ])
                y = net.predict(input_data_path)
                if y is not None:
                    net.save(y, file_info['start'].strftime("%Y%m%d_%H%M%S"))
            else:
                print(f'{file_path} not exist')
