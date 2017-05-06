import cv2
import numpy as np
import os
def file_name(file_dir):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.png':
                L.append(os.path.join(root, file))
    return L
M = cv2.getRotationMatrix2D((640/2,480/2),180,1)

file_list = file_name("C:\\Users\\lenovo\\Desktop\\RMvideo\\real")
for i in file_list:
    print(i)
    ori = cv2.imread(i)
    mat = cv2.warpAffine(ori, M, (640, 480))
    cv2.imwrite(i, mat)