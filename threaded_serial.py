import time
import threading
import serial
import queue
import os
from utils import construct_measurement_dictionary

class Serial_Communication:
    def __init__(self, port, file_reader_writer):
        self.event = threading.Event()
        self.port = port
        self.producer_lock = threading.Lock()
        self.consumer_lock = threading.Lock()
        self.producer_lock.acquire()
        self.threads = []
        self.ser = None
        self.file_reader_writer = file_reader_writer
        self.command = None
        self.neighbor_stack = []

    def read_port(self, q, ser, event):
        while not event.is_set():
            self.producer_lock.acquire()
            q.put(ser.read_until())
            self.consumer_lock.release()


    def send_measure_command(self, ser, event):
        prep_commands = ["AT+CFUN=1", "AT+CFUN=?", "AT+CFUN=1"]
        prep_commands_idx = 0
        measurement_period = 1
        prep_period = 0.5
        while not event.is_set():
            if prep_commands_idx < len(prep_commands):
                command = prep_commands[prep_commands_idx]
                time.sleep(prep_period)
                prep_commands_idx += 1
            else:
                # command = "AT%NCELLMEAS"
                time.sleep(measurement_period)
            self.consumer_lock.acquire()
            if self.command is not None:
                ser.write(bytes(self.command + "\r\n", 'utf-8'))
                self.command = None
            self.producer_lock.release()


    def show_data(self, q, event):
        while not event.is_set():
            while not q.empty():
                message = q.get()
                message = message[:-2]
                message = str(message, 'utf-8')
                self.file_reader_writer.write(message)
                print("Show data:", message)


    def evaluate(self, command, response):
        if command == "AT+CFUN=1" and response == "OK":
            return "AT%NCELLMEAS"
        elif command == "AT%NCELLMEAS" and response.find("%NCELLMEAS") > 0:
            if response.find("%NCELLMEAS: 0") > 0:
                current_measurement_dictionary = construct_measurement_dictionary(response)
                # print("3", current_measurement_dictionary)
                if "neighbor_cells" in current_measurement_dictionary:
                    self.neighbor_stack = current_measurement_dictionary["neighbor_cells"]
            return "AT+CFUN=0"
        elif command == "AT+CFUN=0" and response == "OK":


    def initialize(self):
        self.ser = serial.Serial(self.port, 115200, timeout=None)
        q = queue.Queue()
        thread1 = threading.Thread(target=self.read_port, args=(q, self.ser, self.event))
        thread2 = threading.Thread(target=self.show_data, args=(q,self.event))
        thread3 = threading.Thread(target=self.send_measure_command, args=(self.ser,self.event))
        thread1.start()
        thread2.start()
        thread3.start()
        self.threads.append(thread1)
        self.threads.append(thread2)
        self.threads.append(thread3)


    def close_connection(self):
        print("closing")
        self.event.set()
        for thread in self.threads:
            thread.join()

        self.ser.close()



class File_Reader_Writer:
    def __init__(self, filename):
        self.producer_lock = threading.Lock()
        self.consumer_lock = threading.Lock()
        self.consumer_lock.acquire()
        self.filename = filename

    def write(self, message):

        print(message)

        if message.find("%NCELLMEAS: 0") >= 0:
            print("MEssage:", message)
            # _, measurement_list = construct_measurement_dictionary(message, return_measurement_list=True)
            self.producer_lock.acquire()
            with open(self.filename, "a+") as file:
                file.write(message + "\n")
            self.producer_lock.release()

    def read(self):
        self.producer_lock.acquire()
        with open(self.filename) as file:
            measurement_lines = file.readlines()
        self.producer_lock.release()
        return measurement_lines

if __name__ == '__main__':
    # ser = serial.Serial("COM4", 115200, timeout=None)
    # q = queue.Queue()
    # ser_com = Serial_Communication()
    # thread1 = threading.Thread(target=ser_com.read_port, args=(q, ser), daemon=True)
    # thread2 = threading.Thread(target=ser_com.show_data, args=(q,), daemon=True)
    # thread3 = threading.Thread(target=ser_com.send_measure_command, args=(ser, ), daemon=True)
    #
    # thread1.start()
    # thread2.start()
    # thread3.start()
    #
    # thread1.join()
    # thread2.join()
    # thread3.join()
    csv_reader_writer = File_Reader_Writer("saved_measurements/1.txt")
    ser_com = Serial_Communication("COM4", csv_reader_writer)
    ser_com.initialize()


    # csv_reader_writer.read_csv()

    time.sleep(5)
    print("closing connection")
    ser_com.close_connection()
    print("closed connection")
    time.sleep(5)
