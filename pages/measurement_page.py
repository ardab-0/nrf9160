import streamlit as st
from threaded_serial import File_Reader_Writer, Serial_Communication
from utils import construct_measurement_dictionary, query_base_station_dataset, load_data, \
    calculate_timing_advance_distance
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
import numpy as np
from triangulation import multilateration
from geodesic_calculations import get_cartesian_coordinates, get_coordinates, get_distances_in_cartesian, get_distance_and_bearing
import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components


MEASUREMENT_UNCERTAINTY = 78.125

base_station_df = load_data("./262.csv")

file_reader_writer = File_Reader_Writer("./saved_measurements/measurements.json")
measurements = file_reader_writer.read()

query_results = []

for measurement_batch in measurements:
    query_result_of_batch_df = pd.DataFrame()
    for i, measurement in enumerate(measurement_batch):
        dictionary = construct_measurement_dictionary(measurement)

        res = query_base_station_dataset(base_station_df, dictionary["plmn"],
                                         dictionary["tac"], dictionary["cell_id"])
        if not res.empty:
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
    if not query_result_of_batch_df.empty:
        query_results.append(query_result_of_batch_df)
    else:
        print("No base station in database in this batch.")

st.write("Database Query Results")
for result_df in query_results:
    st.write(result_df)

# positions_cartesian_coordinates_list = []
# positions_list = []
# multilateration_result_list = []
# distances_list = []

multilateration_result_df = pd.DataFrame()
for result_df in query_results:
    distances = []
    positions = []
    for index, row in result_df.iterrows():
        distances.append(row["distance"])
        positions.append([row["latitude"], row["longitude"]])

    distances = np.array(distances)
    positions = np.array((positions))
    #use first measurement's time for the whole batch
    measurement_time = result_df.iloc[0]["measurement_time"]

    if len(distances) == 1:
        res = pd.DataFrame([{
            'longitude': positions[0, 1],       # order is latitude, longitude in geopy
            'latitude': positions[0, 0],        # but order is longitude, latitude in map
            'std': result_df.iloc[0]["distance"] + MEASUREMENT_UNCERTAINTY,
            'measurement_time': measurement_time
        }])
    else:
        positions_cartesian_coordinates = get_cartesian_coordinates(positions)
        distances_cartesian = get_distances_in_cartesian(distances, positions, positions_cartesian_coordinates)



        multilateration_result = multilateration(distances.T, positions_cartesian_coordinates.T)

        # multilateration_result_list.append(multilateration_result)
        # positions_cartesian_coordinates_list.append(positions_cartesian_coordinates)
        # distances_list.append(distances)
        # positions_list.append(positions)

        orig_coords = np.zeros((2, 2))
        orig_coords[0, :] = positions[0, 0:2]

        triangulated_coords_cartesian = np.zeros((2, 2))
        triangulated_coords_cartesian[1, :] = np.squeeze(multilateration_result[1:])
        triangulated_geographic_coords = get_coordinates(triangulated_coords_cartesian, orig_coords)

        print("Multilaterated results:", triangulated_geographic_coords)
        print("")
        print("Positions cartesian coordiantes: ", positions_cartesian_coordinates)
        print("")
        print("Positions: ", positions)
        print("")
        print("Multilateration result:", triangulated_geographic_coords[1:])
        print("Anchor pos:", positions[1, :])
        print(get_distance_and_bearing(triangulated_geographic_coords[1, :], positions[1, :]))
        print("")
        print("")

        res = pd.DataFrame([{
            'longitude': triangulated_geographic_coords[1, 1],   # order is latitude, longitude in geopy
            'latitude': triangulated_geographic_coords[1, 0],      # but order is longitude, latitude in map
            'std': MEASUREMENT_UNCERTAINTY / np.sqrt(len(result_df)),#############################??????????????????????????????????
            'measurement_time': measurement_time
        }])


    multilateration_result_df = pd.concat([multilateration_result_df, res])


multilateration_result_df = multilateration_result_df.reset_index().drop(columns=["index"])

st.write("Multilateration results")
st.write(multilateration_result_df)

index_of_selected_estimation_result = st.slider("Select time index", 0, len(multilateration_result_df)-1, value=0)

selected_time_df = multilateration_result_df.iloc[index_of_selected_estimation_result].to_frame().T
selected_base_stations_df = query_results[index_of_selected_estimation_result]

# positions_cartesian_coordinates = positions_cartesian_coordinates_list[index_of_selected_estimation_result]
# multilateration_result = multilateration_result_list[index_of_selected_estimation_result]
# distances = distances_list[index_of_selected_estimation_result]
# positions = positions_list[index_of_selected_estimation_result]
#
# # st.write("Cartesian norm:")
# st.write((np.linalg.norm(positions_cartesian_coordinates[0, :] - positions_cartesian_coordinates[1, :])))
# st.write("Geographic distance")
# st.write(get_distance_and_bearing(positions[0, :], positions[1, :]))
# fig, ax = plt.subplots()
# for i in range(len(positions_cartesian_coordinates)):
#     circle = plt.Circle((positions_cartesian_coordinates[i, 0], positions_cartesian_coordinates[i, 1]), distances[i], color='r', fill=False)
#     ax.add_patch(circle)
# ax.set_xlim((-1500, 1500))
# ax.set_ylim((-1500, 1500))
# plt.scatter(multilateration_result[1], multilateration_result[2])
# plt.grid()
# fig_html = mpld3.fig_to_html(fig)
# components.html(fig_html, height=600)


estimation_layer = pdk.Layer(
    "ScatterplotLayer",
    data=selected_time_df,
    pickable=False,
    opacity=0.3,
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

base_station_layer = pdk.Layer(
    "ScatterplotLayer",
    data=selected_base_stations_df,
    pickable=False,
    opacity=0.2,
    stroked=True,
    filled=True,
    line_width_min_pixels=1,
    get_position=["longitude", "latitude"],
    get_radius="distance",
    radius_min_pixels=5,
    radiusScale=1,
    # radius_max_pixels=60,
    get_fill_color=[3, 136, 252],
    get_line_color=[0, 0, 255],
    tooltip="test test",
)

view = pdk.ViewState(latitude=49.5, longitude=11, zoom=10, )
# Create the deck.gl map
r = pdk.Deck(
    layers=[base_station_layer, estimation_layer],
    initial_view_state=view,
    map_style="mapbox://styles/mapbox/light-v10",
)

# Render the deck.gl map in the Streamlit app as a Pydeck chart
st.write("Estimated position(Orange) and Base Station Positions(Blue) at Selected Time Index")
map = st.pydeck_chart(r)

# ################################################ Single Measurement Case #############################################################
# st.title("Single Measurement Case")
#
# distances = np.array([[query_result_of_batch_df.iloc[0]["distance"], query_result_of_batch_df.iloc[1]["distance"]]])
# positions = np.array([[query_result_of_batch_df.iloc[0]["longitude"], query_result_of_batch_df.iloc[0]["latitude"]],
#                       [query_result_of_batch_df.iloc[1]["longitude"], query_result_of_batch_df.iloc[1]["latitude"]]])
#
#
# positions_cartesian_coordinates = get_cartesian_coordinates(positions)
# distances_cartesian = get_distances_in_cartesian(distances, positions, positions_cartesian_coordinates)
#
# multilateration_result = multilateration(distances.T, positions_cartesian_coordinates.T)
# # multilateration_result[1] = (positions_cartesian_coordinates[0,0] + positions_cartesian_coordinates[0,1]) / 2
# # multilateration_result[2] = (positions_cartesian_coordinates[1,0] + positions_cartesian_coordinates[1,1]) / 2
#
#
# orig_coords = np.zeros((2, 2))
# orig_coords[0, :] = positions[0, 0:2]
#
# triangulated_coords_cartesian = np.zeros((2, 2))
# triangulated_coords_cartesian[1, :] = np.squeeze(multilateration_result[1:])
# triangulated_geographic_coords = get_coordinates(triangulated_coords_cartesian,  orig_coords)
#
# geographic_coords = get_coordinates(positions_cartesian_coordinates.T,  positions)
# st.write("Trial")
# st.write(geographic_coords)
#
#
# st.write(distances)
# st.write(positions)
# st.write(triangulated_geographic_coords)
#
# st.write(get_distance_and_bearing((10.9754, 49.4801), (10.9854, 49.4838)))
#
# res = pd.DataFrame([{
#     'longitude': triangulated_geographic_coords[1, 0],
#     'latitude': triangulated_geographic_coords[1, 1],
#     'current_rsrq': -1,
#     'range': -1,
#     'measurement_time': -1,
#     'timing_advance': -1,
#     'distance': 1
# }])
# query_result_of_batch_df = pd.concat([query_result_of_batch_df, res])
#
# st.write(query_result_of_batch_df)
#
# base_station_layer = pdk.Layer(
#     "ScatterplotLayer",
#     data=query_result_of_batch_df,
#     pickable=False,
#     opacity=0.3,
#     stroked=True,
#     filled=True,
#     line_width_min_pixels=1,
#     get_position=["longitude", "latitude"],
#     get_radius="distance",
#     radius_min_pixels=5,
#     # radius_max_pixels=60,
#     get_fill_color=[252, 136, 3],
#     get_line_color=[255, 0, 0],
#     tooltip="test test",
# )
#
# view = pdk.ViewState(latitude=50, longitude=11, zoom=9, )
# # Create the deck.gl map
# r = pdk.Deck(
#     layers=[base_station_layer],
#     initial_view_state=view,
#     map_style="mapbox://styles/mapbox/light-v10",
# )
#
# # Render the deck.gl map in the Streamlit app as a Pydeck chart
# st.write("Base station positions")
# map = st.pydeck_chart(r)
