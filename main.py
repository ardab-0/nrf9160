import serial
import serial.tools.list_ports
import time


# ser = serial.Serial("COM4", 115200)


ports = serial.tools.list_ports.comports()

for port, desc, hwid in sorted(ports):
        print("{}: {} [{}]".format(port, desc, hwid))



# print(ser.read_until(b'The AT host sample started\r\n'))
# time.sleep(5)
# ser.timeout=1
#
#
# ser.write(b'AT+CFUN=1\r\n')
# print("AT+CFUN=1")
# print(ser.read(100))
#
# time.sleep(1)
# ser.write(b'AT+CFUN?\r\n')
# print("AT+CFUN?")
# print(ser.read(100))
#
# time.sleep(0.1)
# ser.write(b'AT%NCELLMEAS\r\n')
# print("AT%NCELLMEAS")
# print(ser.read(100))
#
#
# print("finished")
