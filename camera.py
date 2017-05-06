import cv2
import numpy as np
import tool


cap = cv2.VideoCapture(1)
ret, frame = cap.read()
if ret == False:
    exit(1)
def con(x):
    cap.set(11, x)
    show()
def sat(x):
    cap.set(12, x)
    show()
def hue(x):
    cap.set(13, x - 30)
    show()
def gain(x):
    cap.set(14, x)
    show()
def exp(x):
    cap.set(15, x - 13)
    show()
def show():
    bright = cap.get(10)
    contrast = cap.get(11)
    saturate = cap.get(12)
    hue = cap.get(13)
    gain = cap.get(14)
    exposure = cap.get(15)
    print("bright " + str(bright) + " | contrast " + str(contrast) + " | saturate " + str(saturate) + " | hue " + str(hue) + " | gain " + str(gain) + " | exposure " + str(exposure))


cv2.namedWindow("image")
tool.func_tool_set_quit()

cv2.createTrackbar('Contrast','image',0,32,con)
cv2.createTrackbar('Saturate','image',0,100,sat)
cv2.createTrackbar('Hue','image',0,50,hue)
cv2.createTrackbar('Gain','image',0,100,gain)
cv2.createTrackbar('Exposure','image',0,12,exp)

cv2.setTrackbarPos("Contrast", "image", int(cap.get(11)))
cv2.setTrackbarPos("Saturate", "image", int(cap.get(12)))
cv2.setTrackbarPos("Hue", "image", int(cap.get(13)) + 30)
cv2.setTrackbarPos("Gain", "image", int(cap.get(14)))
cv2.setTrackbarPos("Exposure", "image", int(cap.get(15)) + 13)

while ret:
    ret, frame = cap.read()
    cv2.imshow("image", frame)
    tool.func_tool_set_mouth_callback_show_pix("image", frame)
    cv2.waitKey(10)
    '''
    CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds or video capture timestamp.
    CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.
    CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file: 0 - start of the film, 1 - end of the film.
    CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.
    CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.
    CV_CAP_PROP_FPS Frame rate.
    CV_CAP_PROP_FOURCC 4-character code of codec.
    CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.
    CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .
    CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.
    CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).
    CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).  0 - 32
    CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).  0 - 100
    CV_CAP_PROP_HUE Hue of the image (only for cameras).    -30 - 20
    CV_CAP_PROP_GAIN Gain of the image (only for cameras).  0 - 100
    CV_CAP_PROP_EXPOSURE Exposure (only for cameras).   -13.0 - -1.0
    CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.
    CV_CAP_PROP_WHITE_BALANCE Currently not supported
    CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)
    '''
