import asyncio
import time

import streamlit as st
from datetime import datetime
from threaded_serial import File_Reader_Writer, Serial_Communication

# async def show_measurements(csv_reader_writer):
#     while True:
#         measurement_df = csv_reader_writer.read_csv()
#         st.session_state.measurement_df = measurement_df
#         st.dataframe(st.session_state.measurement_df)
#         r = await asyncio.sleep(3)
#
#
# csv_reader_writer = Csv_Reader_Writer("saved_measurements/1.csv")
# ser_com = Serial_Communication("COM4", csv_reader_writer)
# ser_com.initialize()
# if 'measurement_df' not in st.session_state:
#     st.session_state.measurement_df = None
#
# asyncio.run(show_measurements(csv_reader_writer))
#
# print("closing connection")
# ser_com.close_connection()
# print("closed connection")


file_reader_writer = File_Reader_Writer("saved_measurements/1.txt")
ser_com = Serial_Communication("COM4", file_reader_writer)
ser_com.initialize()
# placeholder = st.empty()
for i in range(200):
    measurements = file_reader_writer.read()
    # with placeholder.container():
    #     st.write(measurements)

    time.sleep(1)
print("closing connection")
ser_com.close_connection()
print("closed connection")

# import pandas as pd
# import streamlit as st
# import numpy as np
#
#
#
# # random value to append; could be a num_input widget if you want
# random_value = np.random.randn()
# if 'data' not in st.session_state:
#     st.session_state.data = pd.DataFrame(columns=["Random"])
#
#
# if st.button("Append random value"):
#     # update dataframe state
#     st.session_state.data = st.session_state.data.append({'Random': random_value}, ignore_index=True)
#     st.text("Updated dataframe")
#     st.dataframe(st.session_state.data)
