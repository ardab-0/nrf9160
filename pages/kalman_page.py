import streamlit as st
from utils import *
from filterpy.kalman import predict, update
from measurement_result import ncellmeas_moving_results, ncellmeas_results
import numpy as np
from geodesic_calculations import get_cartesian_coordinates, get_coordinates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk


def get_kalman_matrices(measurement_sigma=1, dt=1, sigma_a=1):
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


# map_radius_scale = 1 / 1
# sigma_a = 0.03

dataset_type = st.sidebar.selectbox(
     'Dataset type: ',
     ('Stationary', 'Moving'))
sigma_a = st.sidebar.slider('Acceleration Sigma (m/s\u00b2)', 0.0, 2.0, 0.03)
map_radius_scale = st.sidebar.slider('Scale of radii', 0.0, 1.0, 0.03)

measurement = None
if dataset_type == "Stationary":
    measurement = ncellmeas_results
elif dataset_type == "Moving":
    measurement = ncellmeas_moving_results
else:
    raise Exception("Invalid value selected")

base_station_df = load_data("262.csv")
moving_measurement_dictionary_list = get_measurement_dictionary_list(measurement)
moving_path_df = get_moving_path_df(base_station_df, moving_measurement_dictionary_list)

measurements = moving_path_df[["Longitude", "Latitude", "measurement_time", "Range"]].to_numpy()
cartesian_coordinates = get_cartesian_coordinates(measurements[:, 0:2])

x = np.array([[0, 0, 0, 0, 0, 0]], dtype=float).T
filtered_cartesian_coordinates = np.zeros((len(measurements), 2))
P = np.eye(6) * moving_path_df["Range"].iloc[0]
prev_time_ms = 0

measurement_uncertainties = np.zeros((len(measurements), 1))

for i, measurement in enumerate(measurements):
    z = cartesian_coordinates[i, :]
    time_diff_ms = int(measurement[2]) - prev_time_ms
    prev_time_ms = int(measurement[2])
    time_diff_s = int(time_diff_ms / 1000)
    measurement_sigma = int(measurement[3])

    F, H, R, Q = get_kalman_matrices(measurement_sigma, time_diff_s, sigma_a)

    x, P = predict(x, P, F, Q)
    x, P = update(x, P, z, R, H)
    filtered_cartesian_coordinates[i] = np.squeeze(H @ x)
    measurement_uncertainties[i] = P[0, 0]
    # st.write(P)

filtered_geographic_coordinates = get_coordinates(filtered_cartesian_coordinates, measurements[:, 0:2])
moving_path_df["filtered_latitude"] = filtered_geographic_coordinates[:, 1]
moving_path_df["filtered_longitude"] = filtered_geographic_coordinates[:, 0]
moving_path_df["filtered_measurement_uncertainties"] = measurement_uncertainties

# scale map representations of radii to make them more legible in map
moving_path_df["filtered_measurement_uncertainties"] *= map_radius_scale
moving_path_df["Range"] *= map_radius_scale

st.write(moving_path_df)
st.write("Unfiltered Cartesian Coordinates")
st.write(cartesian_coordinates)

st.write("Filtered Cartesian Coordinates")
st.write(filtered_cartesian_coordinates)

st.write("Filtered Geographic Coordinates")
st.write(filtered_geographic_coordinates)

trace1 = go.Scatter(x=cartesian_coordinates[:, 0], y=cartesian_coordinates[:, 1], name="Unfiltered")
trace2 = go.Scatter(x=filtered_cartesian_coordinates[:, 0], y=filtered_cartesian_coordinates[:, 1], name="Filtered")

fig = make_subplots()
fig.add_trace(trace1)
fig.add_trace(trace2)
fig['layout'].update(height=600, width=800, title="Kalman Filter in Cartesian Coordinates", xaxis=dict(title='x (m)'),
                     yaxis=dict(title='y (m)'))
st.plotly_chart(fig)

unfiltered_path_layer = pdk.Layer(
    "ScatterplotLayer",
    data=moving_path_df,
    pickable=False,
    opacity=0.3,
    stroked=True,
    filled=True,
    line_width_min_pixels=1,
    get_position=["Longitude", "Latitude"],
    get_radius="Range",
    radius_min_pixels=5,
    # radius_max_pixels=60,
    get_fill_color=[252, 136, 3],
    get_line_color=[255, 0, 0],
    tooltip="test test",
)

filtered_path_layer = pdk.Layer(
    "ScatterplotLayer",
    data=moving_path_df,
    pickable=False,
    opacity=0.3,
    stroked=True,
    filled=True,
    line_width_min_pixels=1,
    get_position=["filtered_longitude", "filtered_latitude"],
    get_radius="filtered_measurement_uncertainties",
    radius_min_pixels=5,
    # radius_max_pixels=60,
    get_fill_color=[3, 136, 252],
    get_line_color=[0, 0, 255],
    tooltip="test test",
)

view = pdk.ViewState(latitude=50, longitude=10, zoom=3, )
# Create the deck.gl map
r = pdk.Deck(
    layers=[unfiltered_path_layer, filtered_path_layer],
    initial_view_state=view,
    map_style="mapbox://styles/mapbox/light-v10",
)

# Render the deck.gl map in the Streamlit app as a Pydeck chart
map = st.pydeck_chart(r)
