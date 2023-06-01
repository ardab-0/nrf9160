from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import random

max_values = {
    "phys_cell_id": 503,
    "rsrp": 255,  # -17
    "rsrq": 255  # -30
}

min_values = {
    "phys_cell_id": 0,
    "rsrp": -17,
    "rsrq": -30
}


class LSTMDataset(Dataset):
    def __init__(self, x_directory, y_directory, num_features, num_prev_steps,
                 rnd_seed=101):
        np.random.seed(rnd_seed)
        self.num_prev_steps = num_prev_steps

        self.x_df = pd.read_csv(x_directory)
        self.y_df = pd.read_csv(y_directory)

        self.num_features = num_features

        if "original_index" in self.x_df.columns:
            self.augmentation_coefficient = len(self.x_df[self.x_df["original_index"] == 0]) - 1
        else:
            self.augmentation_coefficient = 0

        # # normalize columns
        # for col_name in self.x_df.columns:
        #     if col_name.find("phys_cell_id") >= 0:
        #         self.x_df[col_name] /= (max_values["phys_cell_id"] - min_values["phys_cell_id"])
        #     if col_name.find("rsrp") >= 0:
        #         self.x_df[col_name] /= (max_values["rsrp"] - min_values["rsrp"])
        #     if col_name.find("rsrq") >= 0:
        #         self.x_df[col_name] /= (max_values["rsrq"] - min_values["rsrq"])
        #
        # print("here")
        # train_min = self.x_train_df.min()
        # train_max = self.x_train_df.max()
        # self.x_train_df = (self.x_train_df-train_min)#/(train_max-train_min)
        # self.x_test_df = (self.x_test_df-train_min)#/(train_max-train_min)

    def __len__(self):
        return int(len(self.x_df) / (self.augmentation_coefficient + 1) - self.num_prev_steps + 1)

    def __getitem__(self, idx):

        idx = idx + self.num_prev_steps - 1

        x = self.get_previous_steps_randomly(self.x_df, idx)

        y = self.y_df.iloc[idx, 2]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def get_previous_steps_randomly(self, df, idx):

        x = np.zeros((self.num_prev_steps, self.num_features))
        for i in range(self.num_prev_steps):
            augmentation_idx = random.randint(0, self.augmentation_coefficient)

            x[i] = df[df["original_index"] == (idx - i)].iloc[augmentation_idx, :self.num_features]

        return x
