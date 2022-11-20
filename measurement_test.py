import asyncio
import time

import streamlit as st
from datetime import datetime
from threaded_serial import File_Reader_Writer, Serial_Communication

file_reader_writer = File_Reader_Writer("./saved_measurements/measurements.json")
ser_com = Serial_Communication("COM4", file_reader_writer)
ser_com.initialize()
time.sleep(3000)
print("closing connection")
ser_com.close_connection()
print("closed connection")


