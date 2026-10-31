import numpy as np
import os
from torch.utils.data import Dataset


class salt_dataset(Dataset):
    def __init__(self, data_dir, train=True):
        self.data_dir  = data_dir
        self.data_list = os.listdir(data_dir)
        self.train     = train

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = np.load(
            os.path.join(self.data_dir, self.data_list[index]))

        cube  = data['data']
        label = data['label']


        cube  = cube[None, ...].astype(np.float32)
        label = label[None, ...].astype(np.float32)

        return cube, label
