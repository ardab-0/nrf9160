from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch


class MeasurementDataset(Dataset):
    def __init__(self, x_directory, y_directory, x_normalization,num_features, rnd_seed=101):
        np.random.seed(rnd_seed)
        self.x_directory = x_directory
        self.x_df = pd.read_csv(x_directory)
        self.y_directory = y_directory
        self.y_df = pd.read_csv(y_directory)
        self.num_features = num_features

        # self.x_df = (self.x_df - x_normalization[0]) / x_normalization[1]


    def __len__(self):
        return len(self.x_df)

    def __getitem__(self, idx):
        x = self.x_df.iloc[idx, :self.num_features].to_numpy()
        y = self.y_df.iloc[idx, 2]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
