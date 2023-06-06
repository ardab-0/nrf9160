from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import random
from geodesic_calculations import point_at


class MeasurementDataset(Dataset):
    def __init__(self, x_directory, y_directory, num_features, num_prev_steps, augmentation_count,
                 augmentation_distance_m, is_training, training_set_min_max=None,
                 rnd_seed=101):

        """

        :param x_directory:
        :param y_directory:
        :param num_features: number of features used during training (first num_features columns are used during training)
        :param num_prev_steps: number of previous time steps used for training (appended to end and returned as a single row)
        :param augmentation_count: number of possible augmentation directions
        :param augmentation_distance_m: distance to augment in meters
        :param is_training:
        :param training_set_min_max: (min, max) min and max of training dataset, must be entered if test dataset is used
        :param rnd_seed:
        """
        np.random.seed(rnd_seed)

        self.augmentation_count = augmentation_count
        self.augmentation_distance_m = augmentation_distance_m
        self.num_prev_steps = num_prev_steps
        self.num_features = num_features
        self.is_training = is_training
        self.x_df = pd.read_csv(x_directory)
        self.y_df = pd.read_csv(y_directory)
        self.dataset_len = len(self.x_df)
        self.x_df = self.x_df.iloc[:self.dataset_len, :self.num_features]
        self.y_df = self.y_df.iloc[:self.dataset_len, :]
        self.x_min_max = None
        if is_training:
            x_min = self.x_df.min()
            x_max = self.x_df.max()
            self.x_min_max = (x_min, x_max)
            self.x_df = (self.x_df-x_min)/(x_max-x_min)
        else:
            x_min, x_max = training_set_min_max
            self.x_df = (self.x_df - x_min) / (x_max - x_min)
        # self.x_test_df = (self.x_test_df-train_min)#/(train_max-train_min)

    def __len__(self):
        return self.dataset_len - self.num_prev_steps

    def __getitem__(self, idx):
        idx = idx + self.num_prev_steps  # to account for previous steps

        x = self.x_df.iloc[idx - self.num_prev_steps: idx]

        if self.augmentation_count > 0:
            x = self.randomly_augment_positions(x)

        y = self.y_df.iloc[idx, 2]  # idx col
        return torch.tensor(x.to_numpy().reshape((-1)), dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def get_training_min_max(self):
        if self.is_training:
            return self.x_min_max
        return None

    def randomly_augment_positions(self, df):
        """
        doesn't work, must implement grid division for each time step
        :param df:
        :return:
        """
        for idx, row in df.iterrows():
            rotation_deg = float(np.random.randint(self.augmentation_count)) / self.augmentation_count * 360
            lat = row["latitude"]
            lon = row["longitude"]
            destination_point = point_at((lat, lon), self.augmentation_distance_m, rotation_deg)

            # set randomly augmented coordinates
            row["latitude"] = destination_point[0]
            row["longitude"] = destination_point[1]

        return df
