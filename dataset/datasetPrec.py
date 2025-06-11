import numpy as np
import torch.utils.data as data
from jacksung.utils.multi_task import MultiTasks
from jacksung.utils.cache import Cache


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
