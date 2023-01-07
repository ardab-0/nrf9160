import glob

import streamlit as st
from threaded_serial import File_Reader_Writer, Serial_Communication
from utils import construct_measurement_dictionary, get_base_station_data_web, load_data, \
    calculate_timing_advance_distance, query_base_station_dataset
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
import numpy as np
from triangulation import multilateration
from geodesic_calculations import get_cartesian_coordinates, get_coordinates, get_distances_in_cartesian, \
    get_distance_and_bearing
from utils import get_kalman_matrices
from filterpy.kalman import predict, update
from constants import S_TO_MS

MEASUREMENT_UNCERTAINTY_STEP = 78.125
# orig_position = {"latitude": 49.480269, "longitude": 10.975543}
# orig_position = {"latitude": 49.572990, "longitude": 11.026701}


measurement_filenames = glob.glob("./saved_measurements/*.json")
######################################### Sidebar ###########################################

sigma_a = st.sidebar.slider('Acceleration Standard Deviation (m/s\u00b2)', 0.0, 0.5, 0.01, step=0.01)
measurement_uncertainty = st.sidebar.slider('Measurement Uncertainty (m)', MEASUREMENT_UNCERTAINTY_STEP, 5 * MEASUREMENT_UNCERTAINTY_STEP,
                                            3 * MEASUREMENT_UNCERTAINTY_STEP, step=MEASUREMENT_UNCERTAINTY_STEP)
measurement_filename = st.sidebar.selectbox("Select file to load", measurement_filenames)

db_type = st.sidebar.radio("Database type", ('Online', 'Offline'))

######################################### Sidebar ###########################################

base_station_df = load_data("./262.csv")

file_reader_writer = File_Reader_Writer(measurement_filename)
measurements, orig_position = file_reader_writer.read(get_orig_pos=True)

query_results = []

for measurement_batch in measurements:
    query_result_of_batch_df = pd.DataFrame()
    for i, measurement in enumerate(measurement_batch):
        dictionary = construct_measurement_dictionary(measurement)

        if db_type == "Online":
            res = get_base_station_data_web(dictionary["plmn"],
                                         dictionary["tac"], dictionary["cell_id"])
        else:
            res = query_base_station_dataset(base_station_df, dictionary["plmn"],
                                            dictionary["tac"], dictionary["cell_id"])
        if not res.empty and not dictionary["timing_advance"] == "65535":  # 65535 means timing advance is invalid
            res = pd.DataFrame([{
                'longitude': res["Longitude"].item(),
                'latitude': res["Latitude"].item(),
                'current_rsrq': int(dictionary["current_rsrq"]),
                'range': res["Range"].item(),
                'measurement_time': int(dictionary["measurement_time"]),
                'timing_advance': int(dictionary["timing_advance"]),
                'distance': calculate_timing_advance_distance(round(int(dictionary["timing_advance"]) / 16))
            }])
            query_result_of_batch_df = pd.concat([query_result_of_batch_df, res])

        else:
            print("Base station is not found in database or measurement timing advance is invalid.")

    if not query_result_of_batch_df.empty:
        query_results.append(query_result_of_batch_df)
    else:
        print("No base station in database in this batch.")

with st.expander("Database Query Results"):
    for result_df in query_results:
        st.write(result_df)

multilateration_result_df = pd.DataFrame()

for result_df in query_results:
    distances = []
    positions = []
    for index, row in result_df.iterrows():
        distances.append(row["distance"])
        positions.append([row["latitude"], row["longitude"]])

    distances = np.array(distances)
    positions = np.array((positions))
    # use first measurement's time for the whole batch
    measurement_time = result_df.iloc[0]["measurement_time"]

    if len(distances) == 1:
        res = pd.DataFrame([{
            'longitude': positions[0, 1],  # order is latitude, longitude in geopy
            'latitude': positions[0, 0],  # but order is longitude, latitude in map
            'std': result_df.iloc[0]["distance"] + measurement_uncertainty,
            'measurement_time': measurement_time
        }])
    else:
        positions_cartesian_coordinates = get_cartesian_coordinates(positions)
        distances_cartesian = get_distances_in_cartesian(distances, positions, positions_cartesian_coordinates)

        multilateration_result = multilateration(distances.T, positions_cartesian_coordinates.T)

        orig_coords = np.zeros((2, 2))
        orig_coords[0, :] = positions[0, 0:2]

        triangulated_coords_cartesian = np.zeros((2, 2))
        triangulated_coords_cartesian[1, :] = np.squeeze(multilateration_result[1:])
        triangulated_geographic_coords = get_coordinates(triangulated_coords_cartesian, orig_coords)

        res = pd.DataFrame([{
            'longitude': triangulated_geographic_coords[1, 1],  # order is latitude, longitude in geopy
            'latitude': triangulated_geographic_coords[1, 0],  # but order is longitude, latitude in map
            'std': measurement_uncertainty,  # / np.sqrt(len(result_df)),
            #############################??????????????????????????????????
            'measurement_time': measurement_time
        }])

    multilateration_result_df = pd.concat([multilateration_result_df, res])

multilateration_result_df = multilateration_result_df.reset_index().drop(columns=["index"])

###################################### filter results ######################################
filtered_multilateration_result_df = pd.DataFrame()

measurement_coordinates = multilateration_result_df[["longitude", "latitude"]].to_numpy()
measurement_coordinates_cartesian = get_cartesian_coordinates(measurement_coordinates)

x = np.array([[0, 0, 0, 0, 0, 0]], dtype=float).T
filtered_cartesian_coordinates = np.zeros((len(multilateration_result_df), 2))
P = np.eye(6) * multilateration_result_df["std"].iloc[0]
prev_time_ms = multilateration_result_df["measurement_time"].iloc[0]

measurement_uncertainties = np.zeros((len(multilateration_result_df), 1))

for i, row in multilateration_result_df.iterrows():
    z = measurement_coordinates_cartesian[i, :]
    time_diff_ms = int(row["measurement_time"]) - prev_time_ms
    prev_time_ms = int(row["measurement_time"])
    time_diff_s = int(time_diff_ms / 1000)
    measurement_sigma = int(row["std"])

    F, H, R, Q = get_kalman_matrices(measurement_sigma, time_diff_s, sigma_a)

    x, P = predict(x, P, F, Q)
    x, P = update(x, P, z, R, H)
    filtered_cartesian_coordinates[i] = np.squeeze(H @ x)
    measurement_uncertainties[i] = P[0, 0]
    # st.write(P)

filtered_geographic_coordinates = get_coordinates(filtered_cartesian_coordinates, measurement_coordinates)
filtered_multilateration_result_df["latitude"] = filtered_geographic_coordinates[:, 1]
filtered_multilateration_result_df["longitude"] = filtered_geographic_coordinates[:, 0]
filtered_multilateration_result_df["std"] = measurement_uncertainties
filtered_multilateration_result_df["measurement_time"] = multilateration_result_df["measurement_time"]
###################################### filter results ######################################


col1, col2 = st.columns(2)
with col1:
    st.write("Multilateration results")
    st.write(multilateration_result_df)
with col2:
    st.write("Filtered Multilateration results")
    st.write(filtered_multilateration_result_df)

index_of_selected_estimation_result = st.slider("Select time index", 0, len(multilateration_result_df) - 1, value=0)

selected_time_df = multilateration_result_df.iloc[index_of_selected_estimation_result].to_frame().T
filtered_selected_time_df = filtered_multilateration_result_df.iloc[index_of_selected_estimation_result].to_frame().T

selected_base_stations_df = query_results[index_of_selected_estimation_result]

original_position_layer = pdk.Layer(
    "ScatterplotLayer",
    data=pd.DataFrame(orig_position, index=[0]),
    pickable=False,
    opacity=1,
    stroked=True,
    filled=True,
    line_width_min_pixels=1,
    get_position=["longitude", "latitude"],
    get_radius=5,
    radius_min_pixels=5,
    radiusScale=1,
    # radius_max_pixels=60,
    get_fill_color=[252, 0, 0],
    get_line_color=[255, 0, 0],
    tooltip="test test",
)

estimation_layer = pdk.Layer(
    "ScatterplotLayer",
    data=selected_time_df,
    pickable=False,
    opacity=0.1,
    stroked=True,
    filled=True,
    line_width_min_pixels=1,
    get_position=["longitude", "latitude"],
    get_radius="std",
    radius_min_pixels=5,
    radiusScale=1,
    # radius_max_pixels=60,
    get_fill_color=[252, 136, 3],
    get_line_color=[255, 0, 0],
    tooltip="test test",
)

filtered_estimation_layer = pdk.Layer(
    "ScatterplotLayer",
    data=filtered_selected_time_df,
    pickable=False,
    opacity=0.1,
    stroked=True,
    filled=True,
    line_width_min_pixels=1,
    get_position=["longitude", "latitude"],
    get_radius="std",
    radius_min_pixels=5,
    radiusScale=1,
    # radius_max_pixels=60,
    get_fill_color=[0, 220, 30],
    get_line_color=[0, 255, 0],
    tooltip="test test",
)

base_station_layer = pdk.Layer(
    "ScatterplotLayer",
    data=selected_base_stations_df,
    pickable=False,
    opacity=0.8,
    stroked=True,
    filled=False,
    line_width_min_pixels=1,
    get_position=["longitude", "latitude"],
    get_radius="distance",
    radius_min_pixels=5,
    radiusScale=1,
    # radius_max_pixels=60,
    get_line_color=[0, 0, 255],
    tooltip="test test",
)

base_station_layer_center = pdk.Layer(
    "ScatterplotLayer",
    data=selected_base_stations_df,
    pickable=False,
    opacity=1,
    stroked=True,
    filled=True,
    line_width_min_pixels=1,
    get_position=["longitude", "latitude"],
    get_radius=10,
    radius_min_pixels=5,
    radiusScale=1,
    # radius_max_pixels=60,
    get_line_color=[0, 0, 255],
    tooltip="test test",
)

view = pdk.ViewState(latitude=49.5, longitude=11, zoom=10, )
# Create the deck.gl map
r = pdk.Deck(
    layers=[base_station_layer,base_station_layer_center, estimation_layer, filtered_estimation_layer, original_position_layer],
    initial_view_state=view,
    map_style="mapbox://styles/mapbox/light-v10",
)

# Render the deck.gl map in the Streamlit app as a Pydeck chart
st.write("Estimated position(Orange) and Base Station Positions(Blue) at Selected Time Index")
map = st.pydeck_chart(r)

####################### Uncertainty Graph #################################


trace1 = go.Scatter(x=filtered_multilateration_result_df["measurement_time"] * S_TO_MS,
                    y=filtered_multilateration_result_df["std"], name="filtered measurement uncertainty")
trace2 = go.Scatter(x=multilateration_result_df["measurement_time"] * S_TO_MS, y=multilateration_result_df["std"],
                    name="unfiltered measurement uncertainty")

fig = make_subplots()
fig.add_trace(trace1)
fig.add_trace(trace2)

fig['layout'].update(title="Uncertainty of Filter and Uncertainty of Measurements in Each Time Step",
                     xaxis=dict(title='Measurement Time (s) (difference from modem boot time)'),
                     yaxis=dict(title='Uncertainty / Range (m)'))
st.plotly_chart(fig)
####################### Uncertainty Graph #################################


###################### Difference to Original Position #########################
unfiltered_distances = np.zeros(len(multilateration_result_df))
filtered_distances = np.zeros(len(filtered_multilateration_result_df))

for i, row in multilateration_result_df.iterrows():
    filtered_row = filtered_multilateration_result_df.iloc[i]
    unfiltered_distance, _ = get_distance_and_bearing((orig_position["latitude"], orig_position["longitude"]),
                                                      (row["latitude"], row["longitude"]))
    filtered_distance, _ = get_distance_and_bearing((orig_position["latitude"], orig_position["longitude"]),
                                                    (filtered_row["latitude"], filtered_row["longitude"]))

    unfiltered_distances[i] = unfiltered_distance
    filtered_distances[i] = filtered_distance

trace1 = go.Scatter(x=filtered_multilateration_result_df["measurement_time"] * S_TO_MS,
                    y=filtered_distances, name="filtered distance")
trace2 = go.Scatter(x=multilateration_result_df["measurement_time"] * S_TO_MS, y=unfiltered_distances,
                    name="unfiltered distance")

fig = make_subplots()
fig.add_trace(trace1)
fig.add_trace(trace2)

fig['layout'].update(title="Distance to Original Position in Each Time Step",
                     xaxis=dict(title='Measurement Time (s) (difference from modem boot time)'),
                     yaxis=dict(title='Difference (m)'))
st.plotly_chart(fig)
###################### Difference to Original Position #########################
