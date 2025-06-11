import jacksung.utils.fastnumpy as fnp
import numpy as np
import torch.utils.data as data
from jacksung.utils.cache import Cache

from jacksung.utils.multi_task import MultiTasks


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
