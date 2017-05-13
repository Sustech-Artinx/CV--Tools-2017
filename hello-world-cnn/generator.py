import matplotlib.pyplot as plt
import numpy as np
output = open("label\\label.txt", "w+")

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 320
KERNEL_WIDTH = 40
KERNEL_HEIGHT = 40
theta = 1
position = 0
mul = 1 / np.sqrt(2 * 3.14159) / theta


Gauss_map_y = np.zeros(KERNEL_HEIGHT)
Gauss_map_x = np.zeros(KERNEL_WIDTH)



# 利用 for 循环 实现

for i in range(int(KERNEL_WIDTH / 2)):
    Gauss_map_x[i] = i * 2 / KERNEL_WIDTH
for i in range(int(KERNEL_WIDTH / 2), KERNEL_WIDTH):
    Gauss_map_x[i] = 1 - (i - int(KERNEL_WIDTH / 2)) * 2 / KERNEL_WIDTH

for i in range(KERNEL_HEIGHT):
    Gauss_map_y[i] = i * 2 / KERNEL_HEIGHT
for i in range(int(KERNEL_HEIGHT / 2), KERNEL_HEIGHT):
    Gauss_map_y[i] = 1 - (i - int(KERNEL_WIDTH / 2)) * 2 / KERNEL_HEIGHT

text = []
for c in range(300):
    type = np.random.randint(0, high=2)
    out = np.zeros((IMAGE_WIDTH, IMAGE_HEIGHT))
    if type == 1:
        center_x = np.random.randint(KERNEL_WIDTH, high=IMAGE_WIDTH - KERNEL_WIDTH)
        center_y = np.random.randint(KERNEL_HEIGHT, high=IMAGE_HEIGHT - KERNEL_HEIGHT)
        text.append(str(c) + ".png" + " 1 " + str(center_x) + " " + str(center_y) + "\n")
        for i in range(KERNEL_WIDTH):
            for j in range(KERNEL_HEIGHT):
                out[center_x - int(KERNEL_WIDTH / 2) + i][center_y - int(KERNEL_HEIGHT / 2) + j] = Gauss_map_x[i] * Gauss_map_y[j]
    else:
        text.append(str(c) + ".png 0 0 0\n")
    plt.imsave("data\\" + str(c) + ".png", out, cmap=plt.cm.gray)

output.writelines(text)
output.close();
# 显示和保存生成的图像
# cv2.imshow('raw_img',cv2.resize(out, (300,300)))
# cv2.waitKey(0)
#plt.imsave('out_2.jpg', out, cmap=plt.cm.gray)

