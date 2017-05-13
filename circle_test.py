import cv2
import numpy as np
import tool

'''
This is a 'hello world' test for pantilt test.
Print circle_test.pdf and put the paper close to camera.
'''

def func_detect_circle(frame):
    result = [0,0,0]
#    frame = cv2.resize(frame, (480, 360))
    b,g,r = cv2.split(frame)
    r = cv2.pyrDown(r)
    circles1 = cv2.HoughCircles(r,cv2.HOUGH_GRADIENT,1.2,1200,param1=250,param2=80,minRadius=30,maxRadius=0)
    if np.any(circles1):
        circles = circles1[0,:,:]
        circles = np.uint16(np.around(circles))
        for i in circles[:]:
            if i[2] > result[2]:
                result = list(i)
    return result if sum(result) else []

tool.func_tool_set_quit()
cap = cv2.VideoCapture(1)
while(True):
    ret, frame = cap.read()
    print (func_detect_circle(frame))
