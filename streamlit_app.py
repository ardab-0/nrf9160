import numpy as np
import streamlit as st
import pandas as pd
import pydeck as pdk
import altair as alt
from measurement_result import ncellmeas_results, ncellmeas_moving_results
import plotly.express as px
import numpy as np
from utils import *
import streamlit as st
from utils import *
from filterpy.kalman import predict, update
from measurement_result import ncellmeas_moving_results, ncellmeas_results
import numpy as np
from geodesic_calculations import get_cartesian_coordinates, get_coordinates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pydeck as pdk
from geographiclib.geodesic import Geodesic
import numpy as np
import geopy.distance
import pandas as pd
import streamlit as st
import serial.tools.list_ports
from threaded_serial import Serial_Communication
import queue
import threading

# add session state variables
if 'serial_running' not in st.session_state:
    st.session_state.serial_running = False
if 'ser_com' not in st.session_state:
    st.session_state.ser_com = None

if st.sidebar.button("Refresh COM Ports"):
    st.experimental_rerun()

ports = serial.tools.list_ports.comports()
port_list = []

for port, desc, hwid in sorted(ports):
    # print("{}: {} [{}]".format(port, desc, hwid))
    port_list.append(port)

com_port = st.sidebar.selectbox(
    'COM PORTS: ',
    port_list)

serial_on_off_button = st.sidebar.radio(label='On/Off Connection', options=["off", "on"])

if serial_on_off_button == "on" and st.session_state.serial_running is False:
    st.session_state.ser_com = Serial_Communication(com_port)
    st.session_state.ser_com.initialize()
    st.session_state.serial_running = True
elif serial_on_off_button == "off" and st.session_state.serial_running is True:
    st.session_state.ser_com.close_connection()
    st.session_state.serial_running = False


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
