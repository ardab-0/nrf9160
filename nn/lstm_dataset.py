from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch


class LSTMDataset(Dataset):
    def __init__(self, x_directory, y_directory,num_features, rnd_seed=101):
        np.random.seedfrom torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch


class MeasurementDataset(Dataset):
    def __init__(self, x_directory, y_directory,x_test_directory, y_test_directory, num_features, is_test, rnd_seed=101):
        np.random.seed(rnd_seed)
        self.is_test = is_test

        self.x_train_df = pd.read_csv(x_directory)
        self.y_train_df = pd.read_csv(y_directory)

        self.x_test_df = pd.read_csv(x_test_directory)
        self.y_test_df = pd.read_csv(y_test_directory)
        self.num_features = num_features

        train_min = self.x_train_df.min()
        train_max = self.x_train_df.max()
        self.x_train_df = (self.x_train_df-train_min)/(train_max-train_min)
        self.x_test_df = (self.x_test_df-train_min)/(train_max-train_min)




    def __len__(self):
        return len(self.x_test_df) if self.is_test else len(self.x_train_df)

    def __getitem__(self, idx):

        if self.is_test:
            x = self.x_test_df.iloc[idx, :self.num_features].to_numpy()
            y = self.y_test_df.iloc[idx, 2]
        else:
            x = self.x_train_df.iloc[idx, :self.num_features].to_numpy()
            y = self.y_train_df.iloc[idx, 2]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
(rnd_seed)
        self.x_directory = x_directory
        self.x_df = pd.read_csv(x_directory)
        self.y_directory = y_directory
        self.y_df = pd.read_csv(y_directory)
        self.num_features = num_features

        self.x_df = (self.x_df - self.x_df.min()) / (self.x_df.max() - self.x_df.min())

    def __len__(self):
        return len(self.x_df)

    def __getitem__(self, idx):
        x = self.x_df.iloc[idx, :self.num_features].to_numpy()
        y = self.y_df.iloc[idx, 2]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
