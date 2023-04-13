import time
# from threaded_serial import File_Reader_Writer, Serial_Communication
from single_measurement.threaded_serial import File_Reader_Writer, Serial_Communication


filename = "erlangen_15_02_2023_2.json"


file_reader_writer = File_Reader_Writer("./raw_measurements/" + filename)
ser_com = Serial_Communication("COM4", file_reader_writer)
ser_com.initialize()

time.sleep(30000)

ser_com.close_connection()
print("closed connection")


