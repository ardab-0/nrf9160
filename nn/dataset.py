from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import random


class MeasurementDataset(Dataset):
    def __init__(self, x_directory, y_directory, x_test_directory, y_test_directory, num_features, num_prev_steps,
                 is_test, rnd_seed=101):
        np.random.seed(rnd_seed)
        self.is_test = is_test
        self.num_prev_steps = num_prev_steps

        self.x_train_df = pd.read_csv(x_directory)
        self.y_train_df = pd.read_csv(y_directory)

        self.x_test_df = pd.read_csv(x_test_directory)
        self.y_test_df = pd.read_csv(y_test_directory)
        self.num_features = num_features
        self.augmentation_coefficient = len(self.x_train_df[self.x_train_df["original_index"] == 0]) - 1

        # train_min = self.x_train_df.min()
        # train_max = self.x_train_df.max()
        # self.x_train_df = (self.x_train_df-train_min)#/(train_max-train_min)
        # self.x_test_df = (self.x_test_df-train_min)#/(train_max-train_min)

    def __len__(self):
        return int(len(self.x_test_df) / (
                    self.augmentation_coefficient + 1) - self.num_prev_steps + 1) if self.is_test else int(
            len(self.x_train_df) / (self.augmentation_coefficient + 1) - self.num_prev_steps + 1)

    def __getitem__(self, idx):

        idx = idx + self.num_prev_steps-1


        if self.is_test:
            x = self.get_previous_steps_randomly(self.x_test_df, idx).reshape((-1))
            y = self.y_test_df.iloc[idx, 2]
        else:
            x = self.get_previous_steps_randomly(self.x_train_df, idx).reshape((-1))
            y = self.y_train_df.iloc[idx, 2]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def get_previous_steps_randomly(self, df, idx):

        x = np.zeros((self.num_prev_steps, self.num_features))
        for i in range(self.num_prev_steps):
            augmentation_idx = random.randint(0, self.augmentation_coefficient)

            x[i] = df[df["original_index"] == (idx-i)].iloc[augmentation_idx, :self.num_features]

        return x
