import streamlit as st
from utils import *
from filterpy.kalman import predict, update
from measurement_result import ncellmeas_moving_results
import numpy as np
from geodesic_calculations import get_cartesian_coordinates
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def get_kalman_matrices(measurement_sigma=1, dt=1, sigma_a=0.2):
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

    return F, H, R, Q


base_station_df = load_data("262.csv")
moving_measurement_dictionary_list = get_measurement_dictionary_list(ncellmeas_moving_results)
moving_path_df = get_moving_path_df(base_station_df, moving_measurement_dictionary_list)

measurements = moving_path_df[["Longitude", "Latitude", "measurement_time", "Range"]].to_numpy()
cartesian_coordinates = get_cartesian_coordinates(measurements[:, 0:2])

x = np.array([[0, 0, 0, 0, 0, 0]], dtype=float).T
filtered_cartesian_coordinates = np.zeros((len(measurements), 2))
P = np.eye(6) * 20 ** 2
prev_time_ms = 0

for i, measurement in enumerate(measurements):
    z = cartesian_coordinates[i, :]
    time_diff_ms = int(measurement[2]) - prev_time_ms
    prev_time_ms = int(measurement[2])
    time_diff_s = int(time_diff_ms / 1000)
    measurement_sigma = int(measurement[3])

    F, H, R, Q = get_kalman_matrices(measurement_sigma, time_diff_s)

    x, P = predict(x, P, F, Q)
    x, P = update(x, P, z, R, H)
    filtered_cartesian_coordinates[i] = np.squeeze(H @ x)
    st.write(P)

st.write(moving_path_df)
st.write("Unfiltered Cartesian Coordinates")
st.write(cartesian_coordinates)

st.write("Filtered Cartesian Coordinates")
st.write(filtered_cartesian_coordinates)

trace1 = go.Scatter(x=cartesian_coordinates[:, 0], y=cartesian_coordinates[:, 1], name="Unfiltered")
trace2 = go.Scatter(x=filtered_cartesian_coordinates[:, 0], y=filtered_cartesian_coordinates[:, 1], name="Filtered")

fig = make_subplots()
fig.add_trace(trace1)
fig.add_trace(trace2)
fig['layout'].update(height=600, width=800, title="Kalman Filter in Cartesian Coordinates", xaxis=dict(title='x (m)'), yaxis=dict(title='y (m)'))
st.plotly_chart(fig)

