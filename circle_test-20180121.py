import cv2
import cv2.cv as cv
import numpy as np
import tool
import serial
#import binascill
import string

'''
This is a 'hello world' test for pantilt test.
Print circle_test.pdf and put the paper close to camera.
'''

def sent(t,x,y):
    if x>320:
        yaw_angle = int(3050 - ((x-320)/320 * 3050))
    else:
        yaw_angle = int(3050 + ((320-x)/320 * 2650))
    if y<240:
        pitch_angle = int(5000 - ((240-y)/240 * 5000))
    else:
        pitch_angle = int(5000 + ((y-240)/240 * 1000))
    #t.write(bytes.fromhex(yaw_angle))
    #t.write(bytes.fromhex(pitch_angle))
    mes = bytearray([0,0,0,0,0,0,0,0,0])
    mes[0] = 0xFA
    mes[1] = 0x04
    mes[2] = 0x00
    mes[3] = yaw_angle & 0xFF
    mes[4] = (yaw_angle>>8) & 0xFF
    mes[5] = pitch_angle & 0xFF
    mes[6] = (pitch_angle>>8) & 0xFF
    mes[7] = 0x00
    mes[8] = 0xFB
    print([yaw_angle,pitch_angle])
    t.write(mes)

def func_detect_circle(frame,t):
    result = [0,0,0]
#    frame = cv2.resize(frame, (480, 360))
    b,g,r = cv2.split(frame)
    sp = frame.shape
    #480 640
    r = cv2.pyrDown(r)
    circles1 = cv2.HoughCircles(r,cv.CV_HOUGH_GRADIENT,1.2,1200,param1=250,param2=80,minRadius=30,maxRadius=0)
    if np.any(circles1):
        circles = circles1[0,:,:]
        circles = np.uint16(np.around(circles))
        for i in circles[:]:
            cv2.circle(frame, (2*i[0], 2*i[1]), 2*i[2], (0, 0, 255), 5)
            #i[0] np.uint16
            x = 0.0+2*i[0]
            y = 0.0+2*i[1]
            sent(t,x,y)
            if i[2] > result[2]:
                result = list(i)
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
    #return result if sum(result) else []
    return [2*i[0], 2*i[1]] if sum(result) else []

t = serial.Serial('/dev/ttyUSB0',115200)
tool.func_tool_set_quit()
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    func_detect_circle(frame,t)

    #print (func_detect_circle(frame,t))
