import serial
import time
from utils import construct_measurement_dictionary, get_band



commands = ["AT+CFUN=1", "AT%NCELLMEAS", "AT+CFUN=0", "AT%XCOUNTRYDATA"]
commands_idx = 0
measurement_period = 0.5
current_measurement_dictionary = None
neighbor_stack = []


def write_read(x):
    # print(bytes(x + "\r\n", 'utf-8'))
    ser.write(bytes(x + "\r\n", 'utf-8'))
    time.sleep(0.05)
    data = ser.read_until()
    data = data[:-2]
    data = str(data, 'utf-8')
    return data


ser = serial.Serial(port='COM4', baudrate=115200)
print(ser.read_until(b'The AT host sample started\r\n'))
time.sleep(5)
# ser.timeout = 1
print("Board ready.")

while True:
    if commands_idx < len(commands):
        command = commands[commands_idx]
        if command == "AT%NCELLMEAS":
            response = write_read(command)
            time.sleep(measurement_period)
            print("1", response)
            response = write_read("")
            time.sleep(measurement_period)
            print("1", response)
            commands_idx += 1
            if response.find("%NCELLMEAS: 0") >= 0:
                current_measurement_dictionary = construct_measurement_dictionary(response)
                print("2", current_measurement_dictionary)
                if "neighbor_cells" in current_measurement_dictionary:
                    neighbor_stack = current_measurement_dictionary["neighbor_cells"]
        elif command == "AT%XCOUNTRYDATA":
            if len(neighbor_stack) > 0:
                neighbor = neighbor_stack.pop(0)
                response = write_read('AT%XCOUNTRYDATA=1,"4,{},{},{}"'.format("262", get_band(neighbor["n_earfcn"]), neighbor["n_earfcn"]))
                time.sleep(measurement_period)
                print("3", response)
                commands_idx += 1
            else:
                response = write_read("AT%XCOUNTRYDATA=0")
                time.sleep(measurement_period)
                print("4", response)
                commands_idx = 0

        else:
            response = write_read(command)
            time.sleep(measurement_period)
            print("5", response)
            commands_idx += 1


    else:
        commands_idx = 0
