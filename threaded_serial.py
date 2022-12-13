import time
import threading
import serial
import queue
import os
from fsm import Controller
import json


class Serial_Communication:
    def __init__(self, port, file_reader_writer):
        self.event = threading.Event()
        self.port = port
        self.producer_lock = threading.Lock()
        self.consumer_lock = threading.Lock()
        self.command_producer_lock = threading.Lock()
        self.command_consumer_lock = threading.Lock()
        self.producer_lock.acquire()
        self.command_producer_lock.acquire()
        self.threads = []
        self.ser = None
        self.file_reader_writer = file_reader_writer
        self.command = "AT%XBANDLOCK=0"
        self.neighbor_stack = []
        self.controller = Controller()

    def read_port(self, q, ser, event):
        while not event.is_set():
            self.producer_lock.acquire()
            q.put(ser.read_until())
            self.consumer_lock.release()

    def send_measure_command(self, ser, event):

        # measurement_period = 5
        while not event.is_set():
            # time.sleep(measurement_period)

            self.command_consumer_lock.acquire()
            if self.command is not None:
                print("Send", self.command)
                self.consumer_lock.acquire()
                ser.write(bytes(self.command + "\r\n", 'utf-8'))
                self.producer_lock.release()
                self.command = None
            self.command_producer_lock.release()

    def show_data(self, q, event):
        while not event.is_set():
            while not q.empty():
                message = q.get()
                message = message[:-2]
                message = str(message, 'utf-8')
                print("Show data:", message)
                self.command_producer_lock.acquire()
                self.command, measurement_batch = self.evaluate(message, q)
                self.command_consumer_lock.release()
                if measurement_batch is not None:
                    print("Measurement batch", measurement_batch)
                    if len(measurement_batch) > 0:
                        self.file_reader_writer.write(measurement_batch)


    def evaluate(self, response, q):
        if self.controller.state == "reset_cpsms_before_measure_ok" or self.controller.state == "reset_cpsms_before_measure":
            time.sleep(10)
        else:
            time.sleep(1)

        measurement_batch = None
        if response == "OK" and (self.controller.state != "wait_adjusted_measurement_result"
                                 and self.controller.state != "wait_measurement_result"
                                 and self.controller.state != "measure"
                                 and self.controller.state != "adjusted_measure"):
            self.controller.ok()
        elif response == "OK" and (self.controller.state == "wait_adjusted_measurement_result"
                                 or self.controller.state == "wait_measurement_result"
                                 ):
            return "", measurement_batch
        elif response.find("%NCELLMEAS: 0") >= 0 and (self.controller.state == "wait_adjusted_measurement_result"
                                                      or self.controller.state == "wait_measurement_result"):
            measurement_batch = self.controller.ok(response)[0]

        elif response.find("%NCELLMEAS: 1") >= 0 and self.controller.state == "wait_adjusted_measurement_result":
            self.controller.ok()
            # while not q.empty:
            #     print("Emptying queue due to message order mismatch: ", q.get())
        else:
            time.sleep(1)
            while not q.empty:
                print("Emptying queue due to message order mismatch: ", q.get())
            self.controller.not_ok()
        command = self.controller.get_next_command()[0]
        return command, measurement_batch

    def initialize(self):
        self.ser = serial.Serial(self.port, 115200, timeout=None)
        q = queue.Queue()
        thread1 = threading.Thread(target=self.read_port, args=(q, self.ser, self.event))
        thread2 = threading.Thread(target=self.show_data, args=(q, self.event))
        thread3 = threading.Thread(target=self.send_measure_command, args=(self.ser, self.event))
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

    def write(self, measurement_batch):
        # print(message)
        self.producer_lock.acquire()
        with open(self.filename, 'r+') as file:
            file_data = json.load(file)
            file_data["measurements"].append(measurement_batch)
            file.seek(0)
            json.dump(file_data, file, indent=4)
        self.producer_lock.release()


    def read(self):#######################################################################################################################################################################
        self.producer_lock.acquire()
        with open(self.filename, 'r+') as file:
            file_data = json.load(file)
            measurements = file_data["measurements"]
        self.producer_lock.release()
        return measurements


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
