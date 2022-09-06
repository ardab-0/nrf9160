import streamlit as st
from utils import *
from filterpy.kalman import predict, update
from measurement_result import ncellmeas_moving_results
import numpy as np


def get_kalman_matrices(measurement_sigma=1, dt=1, sigma_a=0.2, initial_p_sigma = 20):
    F = np.array([[1, dt, 0.5 * dt ** 2, 0, 0, 0],
                  [0, 1, dt, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 1, dt, 0.5 * dt ** 2],
                  [0, 0, 0, 0, 1, dt],
                  [0, 0, 0, 0, 0, 1]
                  ], dtype=float)

    H = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0]], dtype=float)

    R = np.array([[measurement_sigma, 0],
                  [0, measurement_sigma]], dtype=float)

    Q = sigma_a ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2, dt ** 2 / 2, 0, 0, 0],
                                 [dt ** 3 / 2, dt ** 2, dt, 0, 0, 0],
                                 [dt ** 2 / 2, dt, 1, 0, 0, 0],
                                 [0, 0, 0, dt ** 4 / 4, dt ** 3 / 2, dt ** 2 / 2],
                                 [0, 0, 0, dt ** 3 / 2, dt ** 2, dt],
                                 [0, 0, 0, dt ** 2 / 2, dt, 1]], dtype=float)
    P = np.eye(6) * initial_p_sigma ** 2

    return F, H, R, Q, P


base_station_df = load_data("262.csv")
moving_measurement_dictionary_list = get_measurement_dictionary_list(ncellmeas_moving_results)
moving_path_df = get_moving_path_df(base_station_df, moving_measurement_dictionary_list)

measurements = moving_path_df[["Longitude", "Latitude", "measurement_time", "Range"]].to_numpy()
x = np.array([[0, 0, 0, 0, 0, 0]], dtype=float).T
filtered_measurements = np.zeros((len(measurements), 2))



for i, measurement in enumerate(measurements):
    z = measurement[0:2]
    time_diff_ms = int(measurement[2])
    measurement_sigma = int(measurement[3])

    F, H, R, Q, P = get_kalman_matrices(measurement_sigma, time_diff_ms)

    x, P = predict(x, P, F, Q)
    x, P = update(x, P, z, R, H)
    filtered_measurements[i] = np.squeeze(H @ x)

st.write(moving_path_df)

st.write(filtered_measurements)
