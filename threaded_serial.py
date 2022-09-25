import time
import threading
import serial
import queue


class Serial_Communication:
    def __init__(self):
        self.producer_lock = threading.Lock()
        self.consumer_lock = threading.Lock()
        self.producer_lock.acquire()

    def read_port(self, q, ser):
        while True:
            self.producer_lock.acquire()
            q.put(ser.read_until())
            self.consumer_lock.release()


    def send_measure_command(self, ser):
        prep_commands = ["AT+CFUN=1", "AT+CFUN=?", "AT+CFUN=1"]
        prep_commands_idx = 0
        measurement_period = 1
        prep_period = 0.5
        while True:
            if prep_commands_idx < len(prep_commands):
                command = prep_commands[prep_commands_idx]
                time.sleep(prep_period)
                prep_commands_idx += 1
            else:
                command = "AT%NCELLMEAS"
                time.sleep(measurement_period)
            self.consumer_lock.acquire()
            ser.write(bytes(command + "\r\n", 'utf-8'))
            self.producer_lock.release()


def show_data(q):
    while True:
        while not q.empty():
            message = q.get()
            print("Show data:", message)


if __name__ == '__main__':
    ser = serial.Serial("COM4", 115200, timeout=None)
    q = queue.Queue()
    ser_com = Serial_Communication()
    thread1 = threading.Thread(target=ser_com.read_port, args=(q, ser))
    thread2 = threading.Thread(target=show_data, args=(q,))
    thread3 = threading.Thread(target=ser_com.send_measure_command, args=(ser, ))

    thread1.start()
    thread2.start()
    thread3.start()

    thread1.join()
    thread2.join()
    thread3.join()