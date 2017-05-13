import cv2
import time

'''
Save image from camera every 1sec. File name is timestamp.
'''

cap = cv2.VideoCapture(0)

if cap.isOpened():
    success, frame = cap. read()
else:
    success = False
cap.set(3, 640)
cap.set(4, 480)

print(str(cap.get(3)) + str(cap.get(4)))
count = 0
while success:
    success, frame = cap.read()

    if count == 10:
        name = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())) + ".png"
        cv2.imwrite(name, frame)
        count = 0
    count += 1

    cv2.imshow("frame", frame)
    cv2.waitKey(10)
