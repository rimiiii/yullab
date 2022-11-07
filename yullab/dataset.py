from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd


class FashionDataset(Dataset):
    def __init__(self, train_filename):
        self.df = pd.read_csv(train_filename)
        self.height = 28
        self.width = 28

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.iloc[idx][0]
        img = np.array(self.df.iloc[idx][1:])
        img = img.reshape(-1, self.height, self.width)
        return torch.FloatTensor(img), label