import time
import threading
import serial
import queue


class Serial_Communication:
    def __init__(self, port):
        self.event = threading.Event()
        self.port = port
        self.producer_lock = threading.Lock()
        self.consumer_lock = threading.Lock()
        self.producer_lock.acquire()
        self.threads = []
        self.ser = None

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
                command = "AT%NCELLMEAS"
                time.sleep(measurement_period)
            self.consumer_lock.acquire()
            ser.write(bytes(command + "\r\n", 'utf-8'))
            self.producer_lock.release()


    def show_data(self, q, event):
        while not event.is_set():
            while not q.empty():
                message = q.get()
                print("Show data:", message)

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
    ser_com = Serial_Communication("COM4")
    ser_com.initialize()

    time.sleep(5)
    print("closing connection")
    ser_com.close_connection()
    print("closed connection")
    time.sleep(5)
