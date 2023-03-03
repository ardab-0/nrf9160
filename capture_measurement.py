import asyncio
import time

import streamlit as st
from datetime import datetime
# from threaded_serial import File_Reader_Writer, Serial_Communication
from single_measurement.threaded_serial import File_Reader_Writer, Serial_Communication


file_reader_writer = File_Reader_Writer("./saved_measurements/erlangen_15_02_2023_2.json")
ser_com = Serial_Communication("COM4", file_reader_writer)
ser_com.initialize()
time.sleep(30000)
print("closing connection")
ser_com.close_connection()
print("closed connection")


