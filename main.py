import serial

with serial.Serial('COM12', 115200, timeout=1) as ser:
    while True:
        line = str(ser.read(200))   # read a '\n' terminated line
        print(line)
        if line.find("The AT host sample started") >=0:
            ser.write(b'AT+CFUN?\r\n')
        if line.find("+CFUN: 0\r\nOK\r\n") >= 0:
            ser.write(b'AT+CFUN=1\r\n')
