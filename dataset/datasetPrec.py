import math
import os
import random
import time
from scipy.ndimage import zoom
import numpy as np
import torch
import torch.utils.data as data
from datetime import datetime, timedelta
from jacksung.utils.multi_task import MultiTasks
from jacksung.utils.time import Stopwatch
import jacksung.utils.fastnumpy as fnp
from tqdm import tqdm
from jacksung.utils.cache import Cache
from util.utils import getFY_coord, make_alt
import threading
import cv2


# era5 mean: [0.1464374302219252, -0.5929449101113802, 288.1234538179886, 101336.55857086771, 0.0001433395789417047]
# era5 std: [5.876842665226308, 4.485294695502531, 14.735881411571976, 1090.3879454112227, 0.0005164300649201773]
#
# era5 mean: [[4007.1058382167334, 3585.318375181981, 3650.6687752798853, 3355.3157611403603,
# 3366.052330804404, 2834.218683376971, 2563.202181057768, 2484.753563132937, 2411.0998968725553],
#           [4005.2099633991847, 3583.449182334927, 3648.820918234433, 3353.575635601469,
# 3364.3373105325327, 2832.041766321419, 2561.024118187693, 2482.5558891691494, 2408.75630638129],
#           [4007.017581031058, 3585.0737578835065, 3650.51636505733, 3355.096622102406,
# 3365.9524667895876, 2833.2992080521326, 2562.1300970284847, 2483.652480219525, 2409.6741125519143],
#           [4007.3178773008644, 3585.243833629111, 3650.862291118841, 3355.449478327296,
# 3366.2521599958995, 2833.575514821367, 2562.308766719239, 2483.7681049444373, 2409.7490679476828]]
#
# era5 std: [[83.18208891726005, 401.8831019785451, 168.42887266931402, 268.04664571447773,
# 259.5613947932753, 527.6999785187876, 542.1695337136915, 523.3391500114378, 420.0066721830552],
#           [119.97192287077304, 409.73985000969174, 185.9727404854999, 277.67535657226375,
# 269.5489181060871, 530.5473341501278, 543.9426216685978, 524.8658233148134, 421.5451123009756],
#           [84.53200451378422, 402.9619083096875, 168.93623693895722, 268.361995897849,
# 259.85555426288096, 527.190636866575, 541.2813941719363, 522.2657022540255, 418.48598601903524],
#           [76.4788695025392, 401.86379008230716, 165.64717520501395, 266.63043709457685,
# 258.04928502268024, 526.5574894678992, 540.8027189355018, 521.8139342829373, 417.9266212089434]]

class Benchmark(data.Dataset):
    def __init__(self, data_dir, dataset_txt_path=None, train=False, repeat=1, batch_size=1, split_size=-1):
        super(Benchmark, self).__init__()
        self.train = train
        self.repeat = repeat
        self.data_dir = data_dir
        self.batch_size = batch_size
        f = open(f'{data_dir}/{dataset_txt_path}', 'r')
        self.file_path = [line.split() for line in f.readlines()]
        if split_size > 0:
            self.file_path = self.file_path[:split_size]
        self.nums_train_set = len(self.file_path)
        self.cache = Cache(20000)
        self.pre_add_data(0, self.batch_size)

    def pre_add_data(self, from_idx, to_idx=None):
        if to_idx is None:
            to_idx = from_idx + 1
        for i in range(from_idx, to_idx):
            for file in self.file_path[i % self.nums_train_set]:
                self.load_data(file)

    def read_data(self, path):
        v = np.load(path).astype(np.float32)
        if path.count('fy') > 0:
            v = v[:, 2:, :, :]
        self.cache.add_key(path, v)
        return v

    def load_data(self, path, wait=False):
        v = self.cache.get_key_in_cache(path)
        if v is None:
            mt = MultiTasks(1)
            mt.add_task(path, self.read_data, (path,))
            self.read_data(path)
            if wait:
                return mt.execute_task(print_percent=False)[path]
            else:
                return mt.execute_task_nowait()
        return v

    def __len__(self):
        if self.train:
            nums = self.nums_train_set * self.repeat
        else:
            nums = self.nums_train_set
        return nums

    def __getitem__(self, idx):
        idx = idx % self.nums_train_set
        qpe_path, fy_path = self.file_path[idx]
        qpe_data = self.load_data(qpe_path, wait=True)
        fy_data = self.load_data(fy_path, wait=True)
        self.pre_add_data(idx + self.batch_size)
        return fy_data, qpe_data


if __name__ == '__main__':
    pass
