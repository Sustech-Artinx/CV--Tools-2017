import cv2
import datetime
import numpy as np
import math
import copy
import tool
BLUE, RED = 0, 1
NEAR, MID, FAR, ULTRA = 0, 1, 2, 3
D, R = 0, 1
CAM, PIC, VID = 0, 1, 2

MODE = D
SRC = PIC
TARGET = BLUE
DIST = NEAR

MIN_STEP = 5

tool.func_tool_set_quit()

# camera undistort matrix
mtx = np.array([[ 544.78014225,    0.        ,  332.28614309],
                [   0.        ,  541.53884466,  241.76573558],
                [   0.        ,    0.        ,    1.        ]])
dist = np.array([[ -4.35436872e-01, 2.13933541e-01, 4.09271605e-04, 5.63531212e-03, -6.74471459e-03]])
def func_undistort(mat):
    """
    undistort a frame
    :param mat: a frame
    :return: undistort frame
    """
    h,w = mat.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    dst = cv2.undistort(mat, mtx, dist, None, newcameramtx)
    return dst

def track(mat, color_threshold, square_ratio, angle_rate, length_rate, matrix_threshold, DIST):
    """
    1. color threshold 
    2. segment (connected component), get bar
    delete untra small and large
    3. match bar, get bar pairs
    similar angle and length
    4. select bar pairs
          |       / 
         |     / |
        |    /  |
       | z /   |
      | /     |  y
     |/    θ|
    --------
        x
    60 < theta < 120
    1.3 < x / y < 3
    :param mat: a frame
    :param color_threshold: 
    :param square_ratio: delete connect component which height / width < square_ratio, delete too "fat"
    :param angle_rate: bar match angle rate factor
    :param length_rate: bar match length rate factor 
    :param matrix_threshold: match pairs by rate score > matrix_threshold
    :param DIST: NEAR or FAR, not use
    :return: [] when no target, [x, y, pixel count] for target
    """
    # Relative threshold
    # set 255 if b - r < color_threshold
    b, g, r = cv2.split(mat)
    b = np.asarray(b, dtype='int32')
    r = np.asarray(r, dtype='int32')
    if TARGET == BLUE:
        b = (b - r) < color_threshold
        b = b.astype(np.uint8) * 255
    elif TARGET == RED:
        # more strict threshold condition for red car
        b = (~(((r - b) > color_threshold) & (g < 100) & (r > 100)))
        b = b.astype(np.uint8) * 255
    if MODE == D:
        cv2.imshow('threshold', b)

    # Label Connected Component
    connect_output = cv2.connectedComponentsWithStats(b, 4, cv2.CV_32S)
    """
    connect_output is a tuple - (int component count, ndarray label map, ndarray connect component info, ndarray unused not clear)
    ndarray label map - use int to label connect component, same int value means one component, size equal to "b"
    connect component info - a n * 5 ndarray to show connect component info, [left_top_x, left_top_y, width, height, pixel number]
    """
    # Delete Component according to Height / Width >= 3
    connect_label = connect_output[1]
    # connect_data = [[leftmost (x), topmost (y), horizontal size, vertical size, total area in pixels], ...]
    connect_data = connect_output[2]
    connect_data[connect_data[:, 0] >= mat.shape[1]] = 0   # ?
    connect_data[connect_data[:, 1] >= mat.shape[0]] = 0   # ?
    if MODE == D:
        print("connected components num: " + str(len(connect_output[2])))

    if MODE == D:
        print("square_scale :" + str(connect_data[:, 3] / connect_data[:, 2]))
        connect_max_index = np.argmax(connect_data[:, 4])
        connect_label_show = copy.deepcopy(connect_label)
        connect_label_show = connect_label_show.astype(np.uint8)
        if connect_max_index != 0:
            connect_label_show[connect_label == connect_max_index] = 0
            connect_label_show[connect_label == 0] = connect_max_index
        connect_label_show = cv2.equalizeHist(connect_label_show)
        for i in range(len(connect_data)):
            cv2.rectangle(connect_label_show, (connect_data[i][0] - 1, connect_data[i][1] - 1),
                          (connect_data[i][0] + connect_data[i][2] + 1, connect_data[i][1] + connect_data[i][3] + 1),
                          155, 1)
            cv2.putText(connect_label_show, str(i), (connect_data[i][0] - 5, connect_data[i][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        0, 1, cv2.LINE_AA)
        cv2.imshow('Component', connect_label_show)

    def func_get_delete_strict_list(connect_data, square_ratio):
        """
        strict delete connect component, used when object is near
        :param connect_data: 
        :param square_ratio: 
        :return: index of connect_data need to delete
        """
        # delete vertical / horizontal < square_scale partitions
        connect_ratio_delete_list = np.where((connect_data[:, 3] / connect_data[:, 2]) < square_ratio)[0]
        # delete area < 30 or > 200
        connect_size_delete_list = np.where((connect_data[:, 4] < 30) | (connect_data[:, 4] > 200))[0]
        connect_delete_list = np.hstack((connect_ratio_delete_list, connect_size_delete_list))

        if MODE == D:
            connect_label_delete_show = copy.deepcopy(connect_label).astype(np.uint8)
            for i in connect_delete_list:
                connect_label_delete_show[connect_label_delete_show == i] = 0
            connect_label_delete_show = cv2.equalizeHist(connect_label_delete_show)
            cv2.imshow('Delete Component', connect_label_delete_show)
        return connect_delete_list

    def func_get_delete_loose_list(connect_data, square_ratio):
        """
        func_get_delete_loose_list seems the same as func_get_delete_strict_list.
        Actually I want to distinguish near and far by calling two diff. function, 
        but later I decided to call track() with diff. square_ratio 
        :param connect_data: 
        :param square_ratio: 
        :return: 
        """
        connect_ratio_delete_list = np.where((connect_data[:, 3] / connect_data[:, 2]) < square_ratio)[0]
        # delete area < 5
        connect_size_delete_list = np.where((connect_data[:, 4] < 30) | (connect_data[:, 4] > 3000))[0]  # why 3000 ?
        connect_delete_list = np.hstack((connect_ratio_delete_list, connect_size_delete_list))

        if MODE == D:
            connect_label_delete_show = copy.deepcopy(connect_label).astype(np.uint8)
            for i in connect_delete_list:
                connect_label_delete_show[connect_label_delete_show == i] = 0
            connect_label_delete_show = cv2.equalizeHist(connect_label_delete_show)
            cv2.imshow('Delete Component', connect_label_delete_show)
        return connect_delete_list

    if DIST == NEAR:
        connect_delete_list = func_get_delete_strict_list(connect_data, square_ratio)
    elif DIST == MID:
        connect_delete_list = func_get_delete_loose_list(connect_data, square_ratio)
    connect_remain = []
    for i in range(len(connect_data)):
        if i not in connect_delete_list:
            connect_remain.append(connect_data[i])


    # no target return [] if all partitions are deleted
    if len(connect_remain) < 2:
        return []

    # Get peak_point points in each light bar
    # This is not a good implement.
    # We can first crop this component from label map according to component info
    # This use something like np.argmin, np.argmax
    bar_peak_point = []
    for i in range(len(connect_remain)):
        top_y = connect_remain[i][1]
        top_x_series = np.where(connect_label[top_y + 1, connect_remain[i][0]:connect_remain[i][0] + connect_remain[i][2]] != 0)[0]
        if len(top_x_series) == 0:
            return []
        n1 = int((np.max(top_x_series) + np.min(top_x_series)) / 2 + connect_remain[i][0])
        down_y = connect_remain[i][1] + connect_remain[i][3] - 1
        down_x_series = np.where(connect_label[down_y - 1, connect_remain[i][0]:connect_remain[i][0] + connect_remain[i][2]] != 0)[0]
        if len(down_x_series) == 0:
            return []
        n2 = int((np.max(down_x_series) + np.min(down_x_series)) / 2 + connect_remain[i][0])
        bar_peak_point.append([n1, top_y, n2, down_y, connect_remain[i][4]])
    # bar_peak_point [[top_left_x, top_left_y, down_right_x, down_right_y, pixel count], ...]
    bar_peak_point = np.array(bar_peak_point)

    if MODE == D:
        for i in range(len(bar_peak_point)):
            cv2.circle(mat, (bar_peak_point[i][0], bar_peak_point[i][1]), 3, (0, 0, 255), -1)
            cv2.circle(mat, (bar_peak_point[i][2], bar_peak_point[i][3]), 3, (0, 0, 255), -1)
        for i in range(len(connect_remain)):
            cv2.putText(mat,str(i), (connect_remain[i][0] - 5, connect_remain[i][1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Calculate bar bar_length, list
    bar_length = np.sqrt(np.power((bar_peak_point[:, 3] - bar_peak_point[:, 1]), 2) + np.power((bar_peak_point[:, 2] - bar_peak_point[:, 0]), 2))
    if MODE == D:
        print("bar_length" + str(bar_length))

    # Calculate bar_length matrix
    # not good implement, should use numpy
    # although we only need upper triangular matrix,
    # but it would be faster is we ignore redundant lower part and use numpy
    matrix_bar_length_diff = []
    for i in range(len(bar_length)):
        temp_one_line = []
        for j in range(len(bar_length)):
            if j <= i:
                temp_one_line.append(0)
            else:
                if abs(bar_length[i] - bar_length[j]) == 0:
                    temp_one_line.append(1)
                else:
                    temp_one_line.append(abs(bar_length[i] - bar_length[j]))
        matrix_bar_length_diff.append(temp_one_line)
    matrix_bar_length_diff = np.array(matrix_bar_length_diff)
    if MODE == D:
        print("matrix_bar_length_diff")
        for i in range(len(matrix_bar_length_diff)):
            print(matrix_bar_length_diff[i])

    # Calculate bar angle
    bar_angle = []
    for i in range(len(bar_peak_point)):
        if bar_peak_point[i][2] - bar_peak_point[i][0] == 0:
            bar_angle.append(90)
        else:
            arc = math.atan(float(bar_peak_point[i][3] - bar_peak_point[i][1]) / float(bar_peak_point[i][2] - bar_peak_point[i][0])) * 180 / math.pi
            arc = arc if arc > 0 else arc + 180
            bar_angle.append(arc)
    if MODE == D:
        print("bar_angle" + str(bar_angle))

    # Calculate angle matrix
    matrix_bar_angle_diff = []
    for i in range(len(bar_angle)):
        temp_one_line = []
        for j in range(len(bar_angle)):
            if j <= i:
                temp_one_line.append(0)
            else:
                temp_one_line.append(abs(bar_angle[i] - bar_angle[j]))
        matrix_bar_angle_diff.append(temp_one_line)
    matrix_bar_angle_diff = np.array(matrix_bar_angle_diff)
    if MODE == D:
        print("matrix_bar_angle_diff")
        for i in range(len(matrix_bar_angle_diff)):
            print(matrix_bar_angle_diff[i])

    # Calculate weighted sum matrix
    matrix_bar_sum_diff = angle_rate * matrix_bar_angle_diff + length_rate * matrix_bar_length_diff
    if MODE == D:
        print("matrix_bar_sum_diff")
        for i in range(len(matrix_bar_sum_diff)):
            print(matrix_bar_sum_diff[i])

    # select pairs by threshold
    matrix_bar_sum_diff_threshold = np.zeros(matrix_bar_sum_diff.shape)
    matrix_bar_sum_diff_threshold[(matrix_bar_sum_diff < matrix_threshold) & (matrix_bar_sum_diff > 0)] = 1
    if len(np.where(matrix_bar_sum_diff_threshold == 1)[0]) == 0:
        return []

    """
           /|
       z /  |
       /    |  y
     /    θ|
    --------
        x
       60 < theta < 120
    """
    # bar pair x distance
    matrix_bar_distence_x = np.zeros(matrix_bar_sum_diff_threshold.shape)
    for i in range(len(matrix_bar_distence_x)):
        for j in range(len(matrix_bar_distence_x)):
            if i < j:
                matrix_bar_distence_x[i][j] = abs(bar_peak_point[i][0] + bar_peak_point[i][2] -
                                                  bar_peak_point[j][0] - bar_peak_point[j][2]) / 2
    matrix_bar_distence_x[matrix_bar_distence_x == 0] = 1
    # bar pair y distance
    matrix_bar_distence_y = np.zeros(matrix_bar_sum_diff_threshold.shape)
    for i in range(len(matrix_bar_distence_y)):
        for j in range(len(matrix_bar_distence_y)):
            if i < j:
                matrix_bar_distence_y[i][j] = abs(bar_peak_point[i][1] + bar_peak_point[j][1] -
                                                  bar_peak_point[i][3] - bar_peak_point[j][3]) / 2
    matrix_bar_distence_y[matrix_bar_distence_y == 0] = 0.1  # a trick

    matrix_bar_distence_z = np.zeros(matrix_bar_sum_diff_threshold.shape)
    for i in range(len(matrix_bar_distence_z)):
        for j in range(len(matrix_bar_distence_z)):
            if i < j:
                matrix_bar_distence_z[i][j] = math.sqrt(pow(bar_peak_point[i][1] - bar_peak_point[j][3], 2) + pow(bar_peak_point[i][0] - bar_peak_point[j][2], 2))
    matrix_bar_arccos = (np.power(matrix_bar_distence_x, 2) + np.power(matrix_bar_distence_y, 2) - np.power(matrix_bar_distence_z, 2)) / 2 / matrix_bar_distence_x / matrix_bar_distence_y
    matrix_bar_arccos[matrix_bar_arccos > 1] = 1
    matrix_bar_arccos[matrix_bar_arccos < -1] = 1
    matrix_bar_arccos = np.arccos(matrix_bar_arccos) * 180 / math.pi
    # 60 < theta < 120
    matrix_bar_arccos_threshold = ((matrix_bar_arccos < 120) & (matrix_bar_arccos > 60)).astype(np.float32)

    # 1.3 < x / y < 3
    matrix_bar_ratio = matrix_bar_distence_x / matrix_bar_distence_y
    matrix_bar_ratio_threshold = ((matrix_bar_ratio < 3) & (matrix_bar_ratio > 1.3)).astype(np.float32)
    matrix_match = matrix_bar_ratio_threshold * matrix_bar_sum_diff_threshold * matrix_bar_arccos_threshold
    if MODE == D:
        i, j = np.where(matrix_match == 1)
        for k in range(len(i)):
            cv2.circle(mat, (bar_peak_point[i[k]][0], bar_peak_point[i[k]][1]), 3, (0, 255, 255), -1)
            cv2.circle(mat, (bar_peak_point[i[k]][2], bar_peak_point[i[k]][3]), 3, (0, 255, 255), -1)
            cv2.circle(mat, (bar_peak_point[j[k]][0], bar_peak_point[j[k]][1]), 3, (0, 255, 255), -1)
            cv2.circle(mat, (bar_peak_point[j[k]][2], bar_peak_point[j[k]][3]), 3, (0, 255, 255), -1)
            cv2.circle(mat, (int((bar_peak_point[j[k]][2] + bar_peak_point[j[k]][0] + bar_peak_point[i[k]][2] + bar_peak_point[i[k]][0]) / 4),
                                     int((bar_peak_point[j[k]][3] + bar_peak_point[j[k]][1] + bar_peak_point[i[k]][3] + bar_peak_point[i[k]][1]) / 4)),
                               5, (255, 255, 0), -1)
        print("matrix_bar_arccos")
        for i in range(len(matrix_bar_arccos)):
            print(matrix_bar_arccos[i])
        print("matrix_bar_arccos_threshold")
        for i in range(len(matrix_bar_arccos_threshold)):
            print(matrix_bar_arccos_threshold[i])
        print("matrix_bar_ratio")
        for i in range(len(matrix_bar_ratio)):
            print(matrix_bar_ratio[i])
        print("matrix_bar_ratio_threshold")
        for i in range(len(matrix_bar_ratio_threshold)):
            print(matrix_bar_ratio_threshold[i])
        print("matrix_bar_sum_diff")
        for i in range(len(matrix_bar_sum_diff)):
            print(matrix_bar_sum_diff[i])
        print("matrix_bar_sum_diff_threshold")
        for i in range(len(matrix_bar_sum_diff_threshold)):
            print(matrix_bar_sum_diff_threshold[i])
        print("matrix_match")
        for i in range(len(matrix_match)):
            print(matrix_match[i])


    if len(np.where(matrix_match == 1)[0]) == 0:
        return []

    # Calculate height sum matrix
    matrix_bar_pixel_sum = np.zeros(matrix_match.shape)
    i, j = np.where(matrix_match == 1)
    for k in range(len(i)):
        matrix_bar_pixel_sum[i[k]][j[k]] = connect_remain[i[k]][4] + connect_remain[j[k]][4]
    if MODE == D:
        print("matrix_bar_height_sum")
        for i in range(len(matrix_bar_pixel_sum)):
            print(matrix_bar_pixel_sum[i])

    # Select Max
    max_i, max_j = np.where(matrix_bar_pixel_sum == np.max(matrix_bar_pixel_sum))
    max_i = max_i[0]
    max_j = max_j[0]
    target_x = int((bar_peak_point[max_j][2] + bar_peak_point[max_j][0] + bar_peak_point[max_i][2] + bar_peak_point[max_i][0]) / 4)
    target_y = int((bar_peak_point[max_j][3] + bar_peak_point[max_j][1] + bar_peak_point[max_i][3] + bar_peak_point[max_i][1]) / 4)
    if MODE == D:
        cv2.circle(mat, (target_x, target_y), 8, (100, 100, 250), -1)
        cv2.circle(mat, (bar_peak_point[max_i][2], bar_peak_point[max_i][3]), 3, (100, 100, 0), -1)
        cv2.circle(mat, (bar_peak_point[max_j][2], bar_peak_point[max_j][3]), 3, (100, 100, 0), -1)
        cv2.circle(mat, (bar_peak_point[max_i][0], bar_peak_point[max_i][1]), 3, (100, 100, 0), -1)
        cv2.circle(mat, (bar_peak_point[max_j][0], bar_peak_point[max_j][1]), 3, (100, 100, 0), -1)
        cv2.imshow('Debug', mat)
        tool.func_tool_set_mouth_callback_show_pix("Debug", mat)
    return np.array([target_x, target_y, bar_peak_point[max_j][2]])



def main(mat):
    global target_last
    global MIN_STEP
    origin = copy.deepcopy(mat)
    tstart = datetime.datetime.now()
    # 80
    if TARGET == BLUE:
        target = track(mat, 80, 2.5, 0.5, 1, 13, NEAR)
        print("Near : Target: x, y " + str(target))
        if len(target) == 0:
            target = track(mat, 80, 0.8, 0.5, 1, 13, MID)
            print("Far : Target: x, y " + str(target))
    elif TARGET == RED:
        target = track(mat, 50, 2, 0.5, 1, 13, NEAR)
        print("Near : Target: x, y " + str(target))
        if len(target) == 0:
            target = track(mat, 50, 0.5, 0.3, 0.6, 13, MID)
            print("Far : Target: x, y " + str(target))
    tend = datetime.datetime.now()
    print(tend - tstart)
    if len(target) != 0:
        MIN_STEP -= 1
        MIN_STEP = 20 if MIN_STEP < 5 else MIN_STEP
        dist = np.sqrt((np.power(target[0] - target_last[0], 2) +
                        np.power(target[1] - target_last[1], 2)))
        if dist > MIN_STEP:
            target_last += ((target[:2] - target_last) / dist * MIN_STEP).astype(np.int32)
        else:
            target_last = target[:2]
        cv2.circle(origin, (target[0], target[1]), 8, (0, 255, 0), -1)
    if len(target) == 0:
        MIN_STEP += 1
        MIN_STEP = 50 if MIN_STEP > 50 else MIN_STEP
    if TARGET == BLUE:
        cv2.circle(origin, (target_last[0], target_last[1]), 20, (0, 0, 255), 2, 15)
        cv2.line(origin, (target_last[0] - 40, target_last[1]), (target_last[0] + 40, target_last[1]), (0, 0, 255), 1)
        cv2.line(origin, (target_last[0], target_last[1] - 40), (target_last[0], target_last[1] + 40), (0, 0, 255), 1)
    elif TARGET == RED:
        cv2.circle(origin, (target_last[0], target_last[1]), 20, (255, 0, 0), 2, 15)
        cv2.line(origin, (target_last[0] - 40, target_last[1]), (target_last[0] + 40, target_last[1]), (255, 0, 0), 1)
        cv2.line(origin, (target_last[0], target_last[1] - 40), (target_last[0], target_last[1] + 40), (255, 0, 0), 1)
#    out.write(origin)
    cv2.imshow('raw_img', origin)
    tool.func_tool_set_mouth_callback_show_pix("raw_img", origin)


target_last = np.array([320, 240], dtype=np.int32)

if SRC == PIC:
    file_list = tool.func_tool_folder("C:\\Users\\lenovo\\Desktop\\blue_near", "png")
    for i in file_list:
        print(i)
        mat = cv2.imread(i)
    #    mat = cv2.imread("C:\\Users\\lenovo\\Desktop\\RMvideo\\real\\red_near\\2017-05-02-21-41-18.png")
        main(mat)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if SRC == CAM:
    cap = cv2.VideoCapture(1)
    if cap.isOpened():
        success, frame = cap.read()
    else:
        success = False
    while success:
        success, mat = cap.read()
        mat = func_undistort(mat)
        main(mat)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if SRC == VID:
#    fourcc = cv2.VideoWriter_fourcc(*'XVID')
#    out = cv2.VideoWriter('test.avi', fourcc, 20.0, (640, 480))
    cap = cv2.VideoCapture('C:\\Users\\lenovo\\Desktop\\RMvideo\\red.avi')
    skip = 0
    count = -1
    wait = 10 if skip == 0 else 0
    while cap.isOpened():
        count += 1
        ret, mat = cap.read()

        if skip == 0:
            print("frame " + str(count))
            main(mat)
            k = cv2.waitKey(wait)
            if k & 0xFF == ord('q'):
 #               out.release()
                break
            if k & 0xFF == ord('p'):
                wait = 0
            if k & 0xFF == ord('c'):
                wait = 10
        else:
            skip -= 1
    cap.release()
    cv2.destroyAllWindows()

