from torch.utils.data import Dataset
import numpy as np
import torch
import pandas as pd
from PIL import Image


class FashionDataset(Dataset):
    def __init__(self, train_filename, train_transforms=None, img_size=(28, 28), num_classes=10,):
        self.df = pd.read_csv(train_filename)
        self.img_size = img_size
        self.num_classes = num_classes
        self.train_transforms = train_transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label_idx = self.df.iloc[idx][0]
        label = np.eye(self.num_classes)[label_idx]

        img = self.df.iloc[idx][1:]
        img = np.array(img).reshape(self.img_size).astype(np.uint8)
        img = Image.fromarray(img)

        if self.train_transforms:
            img = self.train_transforms(img)
        return img, label
