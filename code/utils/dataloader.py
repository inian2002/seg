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

        cube  = data['data']    # (D, H, W)
        label = data['label']   # (D, H, W)  values: 0 / 1 / -1

        # Raw data already preprocessed externally — no normalization here.
        cube  = cube[None, ...].astype(np.float32)   # (1, D, H, W)
        label = label[None, ...].astype(np.float32)  # (1, D, H, W)

        return cube, label
