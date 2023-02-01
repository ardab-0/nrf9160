import pandas as pd
from geodesic_calculations import point_at
import numpy as np


def one_to_many_augmenter(df, distance_m, k):

    augmentation_np = np.zeros((len(df)*k, len(df.iloc[0])))

    for index, row in df.iterrows():
        lat = row["latitude"]
        lon = row["longitude"]
        for i in range(k):
            rotation_deg = float(i) / k * 360
            destination_point = point_at((lat, lon), distance_m, rotation_deg)
            new_row = row.copy()
            new_row["latitude"] = destination_point[0]
            new_row["longitude"] = destination_point[1]
            augmentation_np[index * k + i] = new_row.to_numpy()

    return pd.concat([df, pd.DataFrame(augmentation_np, columns=df.columns)], ignore_index=True)
