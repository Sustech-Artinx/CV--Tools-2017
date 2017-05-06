#!/usr/bin/env python2
# coding: utf-8

import cv2
import glob
import numpy as np

mtx = np.array([[ 544.78014225,    0.        ,  332.28614309],
                [   0.        ,  541.53884466,  241.76573558],
                [   0.        ,    0.        ,    1.        ]])
dist = np.array([[ -4.35436872e-01, 2.13933541e-01, 4.09271605e-04, 5.63531212e-03, -6.74471459e-03]])
def func_undistort(mat):
    h,w = mat.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    dst = cv2.undistort(mat, mtx, dist, None, newcameramtx)
    return dst

#EXAMPLE
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)
while(True):
    ret, frame = cap.read()
    img = func_undistort(frame)
    cv2.imshow('before',frame)
    cv2.imshow('after',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
