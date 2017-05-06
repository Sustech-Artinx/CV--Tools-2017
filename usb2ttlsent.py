import serial
import time

t = serial.Serial('COM4',9600)
#s = bytes('test\n',encoding = "utf8")
#t.write(s)
i=0
while 1:
    s = bytes(str(i),encoding = "utf8")
    i=i+1
    t.write(s)
    print(s)
    time.sleep(0.1)
    
#print t.portstr    
#print n    
#str = t.read(n)    
#print str
