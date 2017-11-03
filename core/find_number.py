import cv2
import numpy as np
import datetime
import os
import copy
import digit_decoder
import tool

R, D = 0, 1
MODE = D
LAST_WIDTH_CENTER = 70
DEFAULT_WIDTH_CENTER = 70
RATIO = 1.6

if MODE == D:
    import matplotlib.pyplot as plt


def show_hist(mat):
    hist, bins = np.histogram(mat.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    plt.plot(cdf_normalized, color='b')
    plt.hist(mat.flatten(), 256, [0, 256], color='r')
    plt.xlim([0, 256])
    plt.legend(('cdf', 'histogram'), loc='upper left')
    plt.show()
'''
    Input a RGB image, return digit image array: [[], [], [] ... []]
'''
def getNumRect(img):
    global LAST_WIDTH_CENTER
    global LAST_WIDTH_CENTER
    global RATIO
    # Threshold and erosion
    ign, thresh = cv2.threshold(cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(thresh,kernel, iterations=1)
    if MODE == D:
        cv2.imshow("thresh", thresh)
        cv2.imshow("erosion", erosion)

    # Find contour
    image, contours, hier = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bound_rect = []
    for i in contours:
        # top-left point x, top-left point y, width, height
        x, y, w, h = cv2.boundingRect(i)
        # Ignore very small rectangle
        if w > 20 and h > 20:
            bound_rect.append([x, y, w, h])
    if MODE == D:
        for i in bound_rect:
            x, y, w, h = i
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, str(x) + " " + str(y) + " " + str(w) + " " + str(h), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imshow("debug", img)

    # threshold rectangles from current_width_center - 8 < width < current_width_center + 8
    # and current_height_center - 8 < width < current_height_center + 8
    def func_select_rect(current_width_center, current_height_center):
        select_rect = []
        for i in bound_rect:
            x, y, w, h = i
            if current_width_center - 8 < w < current_width_center + 8 and current_height_center - 8 < h < current_height_center + 8:
                select_rect.append([x, y, w, h])
        return select_rect

    def fun_d_draw_rect(img, rect, color):
        for i in rect:
            cv2.rectangle(img, (i[0], i[1]), (i[0] + i[2], i[1] + i[3]), color, 2)
        cv2.imshow("debug", img)

    '''
    Check whether rectangles in $select_rect are target or not.
    Use k-mean to check x value and y value
    Fot best cases, 9 points' pattern looks like    o     o     o
                                                    o     o     o
                                                    o     o     o
    So use 3-mean on 9 x value and 9 y value
    $compactness_x and $compactness_y are mean distance to group center after 3-mean
    The smaller the better.
    If they are very large, we can assert there must be some noise
    If len($select_rect) == 9 and have no noise --> return $select_rect
    If len($select_rect) == 9 and have noise --> Ignore, return []
    If len($select_rect) == 10 and have noise --> Try to remove one noise
    First check if 3 group have 3, 3, 4 members
    If not --> return []
    else --> find one noise in the group which has 4 members
    Calculate $matrix_dist between each two points
    The noise must be very far away from other 3 points
    Sum up $matrix_dist alone one axis and find the maximum which should be the noise
    Delete that noise and return
    '''
    def func_check_point(select_rect):
        point_set_x = np.zeros(len(select_rect), dtype=np.float32)
        point_set_y = np.zeros(len(select_rect), dtype=np.float32)
        for i in range(len(select_rect)):
            point_set_x[i] = select_rect[i][0]
            point_set_y[i] = select_rect[i][1]
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
        compactness_x, label_x, centers_x = cv2.kmeans(point_set_x, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        compactness_y, label_y, centers_y = cv2.kmeans(point_set_y, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        if MODE == D:
            print("compactness: x = " + str(compactness_x) + " y = " + str(compactness_y))
        if compactness_x < 200 and compactness_y < 250 and len(select_rect) == 9:
            return select_rect
        elif len(select_rect) == 10:
            label_count_x = np.array([len(np.where(label_x == 0)[0]), len(np.where(label_x == 1)[0]), len(np.where(label_x == 2)[0])])
            label_count_y = np.array(
                [len(np.where(label_y == 0)[0]), len(np.where(label_y == 1)[0]), len(np.where(label_y == 2)[0])])
            if not (len(np.where(label_count_x == 3)[0]) == 2 and len(np.where(label_count_x == 4)[0]) == 1 and len(np.where(label_count_y == 3)[0]) == 2 and len(np.where(label_count_y == 4)[0]) == 1):
                return []

            for i in range(0, 3):
                index_x_temp = np.where(label_x == i)[0]
                if len(index_x_temp) == 4:
                    index_x = index_x_temp
                    break
            for i in range(0, 3):
                index_y_temp = np.where(label_y == i)[0]
                if len(index_y_temp) == 4:
                    index_y = index_y_temp
                    break

            para_x = np.array([point_set_x[index_x] * 4])
            para_y = np.array([point_set_x[index_y] * 4])
            vert_x = np.transpose(para_x)
            vert_y = np.transpose(para_y)
            matrix_dist = np.abs(para_x - vert_x) + np.abs(para_y - vert_y)
            sum = matrix_dist.sum(axis=0)
            index_max = sum.argmax()
            del select_rect[index_x[index_max]]
            return select_rect
        else:
            return []

    '''
    $num_rect stores 9 digit rectangle
    Sort them according to    0   1   2
                              3   4   5
                              6   7   8
    '''
    def func_sort_rect(num_rect):
        position_sum = num_rect[:, 0] + ((num_rect[:, 1]) << 3)
        position_sort_index = np.argsort(position_sum)
        position_sorted = num_rect[position_sort_index]
        if MODE == D:
            for i in range(len(position_sorted)):
                x, y, w, h = position_sorted[i]
                cv2.putText(img, str(i), (x + 10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 100, 255), 1, cv2.LINE_AA)
            cv2.imshow("debug", img)
        return position_sorted

    # From here, try to select 9 targets from all rectangles
    # Threshold rectangle according to width and height
    # $bias_width_center is a dynamic bias, a sliding window -- [70 - 8 ~ 70 + 8], ...
    # [71 - 8 ~ 71 + 8], [69 - 8 ~ 69 + 8]....
    # The width of sliding window is fixed but center value is shifting
    for bias_width_center in range(0, 15):
        current_width_center = LAST_WIDTH_CENTER + bias_width_center
        current_height_center = int(current_width_center / RATIO)
        select_rect = func_select_rect(current_width_center, current_height_center)
        # if more than 9 rectangles guarantee threshold --> call check function
        # else --> continue sliding window
        if len(select_rect) >= 9:
            if MODE == D:
                fun_d_draw_rect(img, select_rect, (255, 0, 0))
            select_rect = func_check_point(select_rect)
            if len(select_rect) != 0:
                break

        current_width_center = LAST_WIDTH_CENTER - bias_width_center
        current_height_center = int(current_width_center / RATIO)
        select_rect = func_select_rect(current_width_center, current_height_center)
        if len(select_rect) >= 9:
            if MODE == D:
                fun_d_draw_rect(img, select_rect, (255, 0, 0))
            select_rect = func_check_point(select_rect)
            if len(select_rect) != 0:
                break
        select_rect = []

    if MODE == D:
        fun_d_draw_rect(img, select_rect, (0, 0, 255))

    if len(select_rect) >= 9:
        # If targets are found, update $LAST_WIDTH_CENTER to reduce cost for sliding window
        LAST_WIDTH_CENTER = current_width_center
        num_rect = np.array(select_rect)
        return func_sort_rect(num_rect)
    else:
        LAST_WIDTH_CENTER = DEFAULT_WIDTH_CENTER
        return []



def func_get_passwd(ori, digit_sorted):

    '''
    $position_sorted is sorted digit rectangle
    get password image according to relative position to $position_sorted[1]
    if any of $up $down $left $right out of boundary, return []
    '''
    def func_get_passwd_mat(ori, position_sorted):
        up = position_sorted[1][1] - int(position_sorted[1][3] * 1.3)
        down = position_sorted[1][1] - int(position_sorted[1][3] * 0.3)
        left = position_sorted[1][0] - int(position_sorted[1][2] * 0.5)
        right = position_sorted[1][0] + position_sorted[1][2] + int(position_sorted[1][2] * 0.4)
        if up > 0 and down > 0 and left > 0 and right > 0:
            if MODE == D:
                cv2.rectangle(img, (left, up), (right, down), (200, 170, 50), 2)
                cv2.imshow("debug", img)
            mat_passwd = ori[up:down, left:right]
            y2 = (right - left) * (position_sorted[0][1] - position_sorted[2][1]) / (position_sorted[2][0] - position_sorted[0][0])
            pts1 = np.float32([[0, 0], [0, down - up], [right - left, int(down - up - y2)]])
            pts2 = np.float32([[0, 0], [0, down - up], [right - left, down - up]])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(mat_passwd, M, (right - left, down - up))
            return dst
        else:
            if MODE == D:
                print("PASSWD_OUT_OF_BOUND")
            return []


    def func_get_passwd_patch(mat_passwd_R):
        mat_passwd_R_ori = cv2.resize(mat_passwd_R, (150, 50))
    #    show_hist(mat_passwd_R)
        mat_passwd_R[mat_passwd_R < 120] = 0
        mat_passwd_R[mat_passwd_R >= 120] = 255
        kernel = np.ones((3, 3), np.uint8)
        close = cv2.morphologyEx(mat_passwd_R, cv2.MORPH_CLOSE, kernel)
        close = cv2.resize(close, (150, 50))
        template = cv2.imread('template.jpg', 0)
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(close, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.5
        loc = np.where(res >= threshold)
        if len(loc[0]) < 5:
            if MODE == D:
                print("PASSWD_MATCH_LOW")
            return []
        '''
        for i in range(len(loc[0])):
            cv2.rectangle(close, (loc[1][i], loc[0][i]), (loc[1][i] + w, loc[0][i] + h), 255, 1)
        cv2.imshow("template", close)
        '''
        axis_x = loc[1].astype(np.float32)
        axis_y_mean = int(np.mean(loc[0]))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        compactness_x, label_x, centers_x = cv2.kmeans(axis_x, 5, None, criteria, 10,
                                                       cv2.KMEANS_RANDOM_CENTERS)
        if MODE == D:
            for i in centers_x:
                i = int(i)
                cv2.rectangle(close, (i, axis_y_mean), (i + w, axis_y_mean + h), 255, 1)
            cv2.imshow("kmean", close)

        matrix_x = np.hstack((centers_x, centers_x, centers_x, centers_x, centers_x))
        matrix_y = np.transpose(matrix_x)
        matrix_dist = np.abs(matrix_x - matrix_y)
        matrix_dist[matrix_dist > 20] = 0
        if len(np.where(matrix_dist != 0)[0]) != 0:
            if MODE == D:
                print("PASSWD_MATCH_LOW")
            return []
        passwd_set = []
        centers_x = np.sort(centers_x, axis=0)
        for i in centers_x:
            mat_passwd_patch_one = mat_passwd_R_ori[axis_y_mean:axis_y_mean + h, int(i):int(i) + w]
    #        show_hist(mat_passwd_patch_one)
            hist = np.histogram(mat_passwd_patch_one, bins=256)[0]
            t = hist[100:].argmax() + 100
            mat_passwd_patch_one[mat_passwd_patch_one > t - 10] = 255
            mat_passwd_patch_one[mat_passwd_patch_one <= t - 10] = 0
            passwd_set.append(mat_passwd_patch_one)
        if MODE == D:
            passwd = np.hstack((passwd_set[0], passwd_set[1], passwd_set[2], passwd_set[3], passwd_set[4]))
            cv2.imshow("passwd_group", passwd)
        return passwd_set

    def func_passwd_decode(passwd_set):
        patch_set = [[6, 2, 14, 10], [1, 9, 9, 17], [13, 9, 21, 17], [6, 16, 14, 24], [1, 23, 9, 31],
                     [13, 23, 21, 31], [6, 31, 14, 39]]
        if MODE == D:
            '''
            d_passwd_set = copy.deepcopy(passwd_set)
            for i in range(len(d_passwd_set)):
                for j in patch_set:
                     cv2.rectangle(d_passwd_set[i], (j[0], j[1]), (j[2], j[3]), 255, 1)
                plt.subplot(1, 5, i + 1)
                plt.imshow(d_passwd_set[i], cmap='gray')
            plt.show()
            '''
        T = 20
        passwd_template = np.array([[T, T, T, 0, T, T, T],
                                    [0, 0, T, 0, 0, T, 0],
                                    [T, 0, T, T, T, 0, T],
                                    [T, 0, T, T, 0, T, T],
                                    [0, T, T, T, 0, T, 0],
                                    [T, T, 0, T, 0, T, T],
                                    [T, T, 0, T, T, T, T],
                                    [T, 0, T, 0, 0, T, 0],
                                    [T, T, T, T, T, T, T],
                                    [T, T, T, T, 0, T, T]], dtype=np.int32)
        passwd_decode = np.zeros(5, dtype=np.int32)
        passwd_prob = np.zeros(7, dtype=np.int32)
        for i in range(len(passwd_set)):
            passwd_one = passwd_set[i]
            passwd_one[passwd_one != 0] = 1
            for j in range(len(patch_set)):
                passwd_prob[j] = np.sum(passwd_one[patch_set[j][1]:patch_set[j][3], patch_set[j][0]:patch_set[j][2]])
                passwd_prob[passwd_prob > T] = T
                passwd_diff = passwd_template - passwd_prob
                passwd_diff[passwd_diff < 0] *= -2
                passwd_diff_sum = np.sum(passwd_diff, axis=1)
                passwd_decode[i] = np.argmin(passwd_diff_sum)
        return passwd_decode

    mat_passwd = func_get_passwd_mat(ori, digit_sorted)
    if len(mat_passwd) == 0:
        return []
    if MODE == D:
        cv2.imshow("passwd_ori", mat_passwd)
    mat_passwd_R = mat_passwd[:, :, 2]
    passwd_set = func_get_passwd_patch(mat_passwd_R)
    if len(passwd_set) == 0:
        return []
    passwd_decode = func_passwd_decode(passwd_set)
    if MODE == D:
        print("passwd: " + str(passwd_decode))
    return passwd_decode

def func_get_num(ori, digit_sorted):
    # Prepare $mat_all_digit_flat and call #decode, which send digit to CNN
    tan_set = np.array([float(digit_sorted[0][1] - digit_sorted[2][1]) / float(digit_sorted[2][0] - digit_sorted[0][0]),
                        float(digit_sorted[3][1] - digit_sorted[5][1]) / float(digit_sorted[5][0] - digit_sorted[3][0]),
                        float(digit_sorted[6][1] - digit_sorted[8][1]) / float(digit_sorted[8][0] - digit_sorted[6][0])])
    tan = np.mean(tan_set)
    mat_digit = []
    if tan > 0.02:
        for i in range(len(digit_sorted)):
            x, y, w, h = digit_sorted[i]
            b = (w - h * tan) / (1 - tan ** 2)
            a = b * tan
            d = h - a
            c = w - b
            pts1 = np.float32([[0, a], [b, 0], [c, h]])
            pts2 = np.float32([[0, 0], [w, 0], [0, h]])
            M = cv2.getAffineTransform(pts1, pts2)
            mat_digit.append(cv2.warpAffine(ori[y:y + h, x:x + w], M, (w, h)))
    elif tan < -0.02:
        tan *= -1
        for i in range(len(digit_sorted)):
            x, y, w, h = digit_sorted[i]
            b = (w * tan * tan - h * tan) / (tan * tan - 1)
            a = b / tan
            d = h - a
            c = w - b
            pts1 = np.float32([[b, 0], [0, a], [c, h]])
            pts2 = np.float32([[0, 0], [0, h], [w, h]])
            M = cv2.getAffineTransform(pts1, pts2)
            mat_digit.append(cv2.warpAffine(ori[y:y + h, x:x + w], M, (w, h)))
    else:
        for i in range(len(digit_sorted)):
            x, y, w, h = digit_sorted[i]
            mat_digit.append(ori[y:y + h, x:x + w])
    '''
    plt.figure(1)
    for i in range(len(mat_digit)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(mat_digit[i], cmap='gray')
    plt.show()
    '''

    mat_all_digit = np.zeros((9, 28, 28), dtype=np.int32)
    for i in range(len(mat_digit)):
        mat_digit_one = t, mat_digit_one = cv2.threshold(cv2.cvtColor(mat_digit[i], cv2.COLOR_BGR2GRAY), 127, 255,
                                                         cv2.THRESH_BINARY)
        mat_all_digit[i] = cv2.resize(mat_digit_one, (28, 28))
    '''
#   plt.ion()
    plt.figure(1)
    for i in range(len(mat_all_digit)):
        plt.subplot(3, 3, i + 1)
        plt.imshow(mat_all_digit[i], cmap='gray')
    plt.show()
#   plt.pause(0.1)
    '''
    mat_all_digit_flat = (255 - mat_all_digit.reshape(9, 784)) / 255
    result_prob = digit.decode(mat_all_digit_flat)

    result = result_prob.argmax(axis=1)
    result_find = np.zeros(9)
    for i in range(1, 10):
        result_find[i - 1] = len(np.where(result == i)[0])
    if np.sum((result_find == 1).astype(np.uint8)) == 9:
        if MODE == D:
            print("digit: " + str(result))
        return result
    else:
        if MODE == D:
            print("DPGIT_DECODE_DUPLICATE" + str(result))
        return []

tool.func_tool_set_quit()
digit = digit_decoder.Digit()
file_list = tool.func_tool_folder("C:\\Users\\lenovo\\Desktop\\RMvideo\\dafu", "png")
for i in file_list:
    print(i)
    img = cv2.imread(i)
    img = cv2.imread("C:\\Users\\lenovo\\Desktop\\RMvideo\\dafu\\2017-04-07-12-01-08.png")

    ori = copy.deepcopy(img)
    cv2.imshow("raw_img", ori)
    tool.func_tool_set_mouth_callback_show_pix("raw_img", ori)

    tstart = datetime.datetime.now()

    digit_sorted = getNumRect(img)
    if len(digit_sorted) != 0:
        passwd_decode = func_get_passwd(ori, digit_sorted)
        digit_num = func_get_num(ori, digit_sorted)

    tend = datetime.datetime.now()
    print(tend - tstart)
    cv2.waitKey()

cv2.destroyAllWindows()
