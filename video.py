import cv2

'''
Video Recorder
'''

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('blue.avi', fourcc, 20.0, (640, 480))
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        out.write(frame)
        cv2.imshow('temp', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
