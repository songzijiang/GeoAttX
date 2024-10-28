import os.path
from datetime import datetime

import jacksung.utils.fastnumpy as fnp
import numpy as np
import torch.utils.data as data
from jacksung.utils.cache import Cache
from jacksung.utils.data_convert import np2tif
from jacksung.utils.multi_task import MultiTasks
from scipy.ndimage import zoom

from util.utils import getFY_coord, make_alt, getFY_coord_min


class Benchmark(data.Dataset):
    def __init__(self, data_dir, dataset_txt_path=None, train=False, repeat=1, batch_size=1, split_size=-1):
        super(Benchmark, self).__init__()
        self.train = train
        self.repeat = repeat
        self.data_dir = data_dir
        self.batch_size = batch_size
        print(f'reading {data_dir}/{dataset_txt_path}...')
        f = open(f'{data_dir}/{dataset_txt_path}', 'r')
        self.file_path = [line.split() for line in f.readlines()]
        if split_size > 0:
            self.file_path = self.file_path[:split_size]
        self.nums_train_set = len(self.file_path)
        self.cache = Cache(100000)
        self.pre_add_data(0, self.batch_size)

    def pre_add_data(self, from_idx, to_idx=None):
        if to_idx is None:
            to_idx = from_idx + 1
        files = []
        for i in range(from_idx, to_idx):
            for file in self.file_path[i % self.nums_train_set]:
                files.append(file)
        self.load_data(files, wait=False)

    def read_data(self, path):
        v = fnp.load(path).copy().astype(np.float32)
        print(f'[TemporaryTag] load {path}, total {len(self.cache.cache_list)}'.ljust(100), end='[TemporaryTag]\r')
        # data_info = path.split(os.path.sep)
        # year, month, day, idx \
        #     = int(data_info[-4]), int(data_info[-3]), int(data_info[-2]), int(data_info[-1].replace('.npy', ''))
        # date = datetime(year, month, day, idx // 4, idx % 4 * 15)
        # print(1)
        # alt = make_alt(self.coord, date.strftime('%Y-%m-%d %H:%M:%S')).astype(np.float32)
        # alt = zoom(alt, 12)
        # v = np.concatenate((alt[np.newaxis, :, :], v), axis=0).astype(np.float16)
        v = v[2:, :, :]
        # v[v < 2000] = 0
        self.cache.add_key(path, v)
        return v

    def load_data(self, paths, wait=False):
        results = dict()
        mt = MultiTasks(3)
        for path in paths:
            v = self.cache.get_key_in_cache(path)
            if v is None:
                # return self.read_data(path)
                mt.add_task(path, self.read_data, (path,))
            else:
                results[path] = v
        if wait:
            r = mt.execute_task(print_percent=False)
            for path in paths:
                if path not in results.keys():
                    results[path] = r[path]
            return [results[path] for path in paths]
        else:
            return mt.execute_task_nowait()

    def __len__(self):
        if self.train:
            nums = self.nums_train_set * self.repeat
        else:
            nums = self.nums_train_set
        return nums

    def __getitem__(self, idx):
        idx = idx % self.nums_train_set
        f_path, n_path, p_path = self.file_path[idx]
        f, n, p = self.load_data([f_path, n_path, p_path], wait=True)
        self.pre_add_data(idx + self.batch_size)
        return f, n, p


if __name__ == '__main__':
    coord = getFY_coord(133)
    coord.h = coord.h // 120
    coord.w = coord.w // 120
    alt = make_alt(coord, datetime.now().strftime('%Y-%m-%d %H:%M:%S')).astype(np.float32)
    print(f'*1*{alt.shape}, {alt[10, 10]}*1*')
    alt = zoom(alt, 12)
    print(f'*2*{alt.shape}, {alt[120, 120]}*1*')
