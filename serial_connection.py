import serial
import time

prep_commands = ["AT+CFUN=1", "AT+CFUN=?", "AT+CFUN=1"]
prep_commands_idx = 0
measurement_period = 1
prep_period = 0.5

def write_read(x):
    print(bytes(x + "\r\n", 'utf-8'))
    ser.write(bytes(x + "\r\n", 'utf-8'))
    # time.sleep(0.05)
    data = ser.read_until()
    return data


ser = serial.Serial(port='COM4', baudrate=115200)
print(ser.read_until(b'The AT host sample started\r\n'))
time.sleep(5)
ser.timeout = 1
print("Board ready.")

while True:
    if prep_commands_idx < len(prep_commands):
        command = prep_commands[prep_commands_idx]
        time.sleep(prep_period)
        prep_commands_idx += 1
    else:
        command = "AT%NCELLMEAS"
        time.sleep(measurement_period)

    response = write_read(command)
    print(response)
