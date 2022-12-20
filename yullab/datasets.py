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


class GrooveDataset(Dataset):
    """Pre-processed Groove dataset."""

    def __init__(self, csv_file, transform):
        """
        Args:
            csv_file (string): Path to the csv file with piano rolls per song.
            transform (callable): Transform to be applied on a sample, is expected to implement "get_sections".
        """
        self.drum_df = pd.read_csv(csv_file, index_col=['filename', 'timestep'])
        self.transform = transform
        self.init_dataset()

    def init_dataset(self):
        """
            Sets up an array containing a pd index (the song name) and the song section,
            ie. [("Song Name:1", 0), ("Song Name:1", 1), ("Song Name:1", 2)]
            for use in indexing a specific section
        """
        indexer = self._get_indexer()

        self.index_mapper = []
        for i in indexer:
            split_count = self.transform.get_sections(len(self.drum_df.loc[i].values))
            for j in range(0, split_count):
                self.index_mapper.append((i, j))


    def __len__(self):
        return len(self.index_mapper)

    def get_mem_usage(self):
        """
            Returns the memory usage in MB
        """
        return self.drum_df.memory_usage(deep=True).sum() / 1024**2

    def _get_indexer(self):
        """
            Get an indexer that treats each first level index as a sample.
        """
        return self.drum_df.index.get_level_values(0).unique()

    def __getitem__(self, idx):
        """
            Our frame is multi-index, so we're thinking each song is a single sample,
            and getting the individual bars is a transform of that sample?
        """
        song_name, section = self.index_mapper[idx]

        sample = self.drum_df.loc[song_name].values
        return sample[section*self.transform.split_size:self.transform.split_size*(section+1)]