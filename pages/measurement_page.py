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


base_station_df = load_data("./262.csv")

file_reader_writer = File_Reader_Writer("./saved_measurements/measurements.json")
measurements = file_reader_writer.read()

query_results = []

for measurement_batch in measurements:
    query_result_of_batch = []
    df = pd.DataFrame()
    for i, measurement in enumerate(measurement_batch):
        dictionary = construct_measurement_dictionary(measurement)

        res = query_base_station_dataset(base_station_df, dictionary["plmn"],
                                         dictionary["tac"], dictionary["cell_id"])
        if not res.empty:
            el = [res["Longitude"].item(), res["Latitude"].item(), int(dictionary["current_rsrq"]), res["Range"].item(),
                  int(dictionary["measurement_time"]), int(dictionary["timing_advance"])]
            query_result_of_batch.append(el)
            res = pd.DataFrame([{
                'Longitude': el[0],
                'Latitude': el[1],
                'timing_advance': el[5],
                'distance': calculate_timing_advance_distance(round(el[5] / 16))
            }])
            df = pd.concat([df, res])
    query_results.append(query_result_of_batch)
    break


distances = np.array([[df.iloc[0]["distance"], df.iloc[1]["distance"]]])
positions = np.array([[df.iloc[0]["Longitude"], df.iloc[0]["Latitude"]],
                      [df.iloc[1]["Longitude"], df.iloc[1]["Latitude"]]])


positions_cartesian_coordinates = get_cartesian_coordinates(positions)
distances_cartesian = get_distances_in_cartesian(distances, positions, positions_cartesian_coordinates)

multilateration_result = multilateration(distances.T, positions_cartesian_coordinates.T)
# multilateration_result[1] = (positions_cartesian_coordinates[0,0] + positions_cartesian_coordinates[0,1]) / 2
# multilateration_result[2] = (positions_cartesian_coordinates[1,0] + positions_cartesian_coordinates[1,1]) / 2


orig_coords = np.zeros((2, 2))
orig_coords[0, :] = positions[0, 0:2]

triangulated_coords_cartesian = np.zeros((2, 2))
triangulated_coords_cartesian[1, :] = np.squeeze(multilateration_result[1:])
triangulated_geographic_coords = get_coordinates(triangulated_coords_cartesian,  orig_coords)

geographic_coords = get_coordinates(positions_cartesian_coordinates.T,  positions)
st.write("Trial")
st.write(geographic_coords)


st.write(distances)
st.write(positions)
st.write(triangulated_geographic_coords)

st.write(get_distance_and_bearing((10.9754, 49.4801), (10.9854, 49.4838)))

res = pd.DataFrame([{
    'Longitude': triangulated_geographic_coords[1, 0],
    'Latitude': triangulated_geographic_coords[1, 1],
    'timing_advance': 5,
    'distance': 1
}])
df = pd.concat([df, res])

st.write(df)

base_station_layer = pdk.Layer(
    "ScatterplotLayer",
    data=df,
    pickable=False,
    opacity=0.3,
    stroked=True,
    filled=True,
    line_width_min_pixels=1,
    get_position=["Longitude", "Latitude"],
    get_radius="distance",
    radius_min_pixels=5,
    # radius_max_pixels=60,
    get_fill_color=[252, 136, 3],
    get_line_color=[255, 0, 0],
    tooltip="test test",
)

view = pdk.ViewState(latitude=50, longitude=11, zoom=9, )
# Create the deck.gl map
r = pdk.Deck(
    layers=[base_station_layer],
    initial_view_state=view,
    map_style="mapbox://styles/mapbox/light-v10",
)

# Render the deck.gl map in the Streamlit app as a Pydeck chart
st.write("Base station positions")
map = st.pydeck_chart(r)
