import numpy as np
import streamlit as st
import pandas as pd
import pydeck as pdk
import altair as alt
from measurement_result import ncellmeas_results, ncellmeas_moving_results
import plotly.express as px
import numpy as np
from utils import *

# Ncellmeas command output
ncellmeas_result = '%NCELLMEAS: 0' \
                   ',"01A37603","26201","57F7",164,6400,167,45,19,102760,' \
                   '1300,364,40,24,24' \
                   ',500,55,27,26,33,' \
                   '100702'

# Load rows of data into the dataframe.
first_n_elements = 10000
base_station_df = load_data("262.csv")
st.write("First {} elements of Open Cell ID Dataset:".format(first_n_elements))
st.write(base_station_df.iloc[:first_n_elements])

st.write("NCELLMEAS measurement result in raw format:")
measurement_dict = construct_measurement_dictionary(ncellmeas_result)
st.write(measurement_dict)

query_results_df = query_base_station_dataset(base_station_df, measurement_dict["plmn"], measurement_dict["tac"],
                                              measurement_dict["cell_id"])
st.write("Base stations measured by NCELLMEAS command (Main base station):")
st.write(query_results_df)

# Set viewport for the deckgl map
view = pdk.ViewState(latitude=50, longitude=10, zoom=2, )
# Create the scatter plot layer
base_station_positions_layer = pdk.Layer(
    "ScatterplotLayer",
    data=query_results_df,
    pickable=False,
    opacity=0.3,
    stroked=True,
    filled=True,
    line_width_min_pixels=1,
    get_position=["Longitude", "Latitude"],
    get_radius="Range",
    radius_min_pixels=5,
    get_fill_color=[252, 136, 3],
    get_line_color=[255, 0, 0],
    tooltip="test test",
)

# Create the deck.gl map
r = pdk.Deck(
    layers=[base_station_positions_layer],
    initial_view_state=view,
    map_style="mapbox://styles/mapbox/light-v10",
)

# Render the deck.gl map in the Streamlit app as a Pydeck chart
map1 = st.pydeck_chart(r)

# measurement signal power chart of current base station
measurement_dictionary_list = get_measurement_dictionary_list(ncellmeas_moving_results)

df = pd.DataFrame(
    np.array([[int(measurement_dict["current_rsrp"]) - 140 for measurement_dict in measurement_dictionary_list],
              [int(measurement_dict["measurement_time"]) / 1000 for measurement_dict in measurement_dictionary_list],
              [int(measurement_dict["current_rsrq"]) / 2 - 19.5 for measurement_dict in
               measurement_dictionary_list]]).T,
    columns=["current_rsrp", "measurement_time", "current_rsrq"])

rsrp_fig = px.line(
    df,
    x="measurement_time",
    y='current_rsrp',
    labels={
        "measurement_time": "Measurement Time (s) (difference from modem boot time)",
        "current_rsrp": "Current RSRP (dBm)"
    },
    title='RSRP - Measurement Time'
)

rsrq_fig = px.line(
    df,
    x="measurement_time",
    y='current_rsrq',
    labels={
        "measurement_time": "Measurement Time (s) (difference from modem boot time)",
        "current_rsrq": "Current RSRQ (dB)"
    },
    title='RSRQ - Measurement Time'
)

st.plotly_chart(rsrp_fig)
st.plotly_chart(rsrq_fig)


moving_measurement_dictionary_list = get_measurement_dictionary_list(ncellmeas_moving_results)
moving_path_df = get_moving_path_df(base_station_df, moving_measurement_dictionary_list)


base_station_positions_layer = pdk.Layer(
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
    radius_max_pixels=60,
    get_fill_color=[252, 136, 3],
    get_line_color=[255, 0, 0],
    tooltip="test test",
)

# Create the deck.gl map
r = pdk.Deck(
    layers=[base_station_positions_layer],
    initial_view_state=view,
    map_style="mapbox://styles/mapbox/light-v10",
)

# Render the deck.gl map in the Streamlit app as a Pydeck chart
map2 = st.pydeck_chart(r)





