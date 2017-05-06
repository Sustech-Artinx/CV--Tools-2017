import serial
import time

ser = serial.Serial('COM3',9600)
data = ''
while 1:
    data = ser.read(ser.inWaiting())
        #if data is not '': 
    print(data)
    #time.sleep(1)
