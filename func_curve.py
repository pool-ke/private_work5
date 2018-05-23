import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import math

iter_count=10
ROI_count=2
img_count=10
ROI_weight=[1,1,1]
min_array=[]
mean_array=[]
max_array=[]
var_array=[]
std_array=[]
for t in range(iter_count):
    str_t=str(t)
    goal_sum_array=[]
    for i in range(img_count):
        str_i=str(i)
        goal_sum=0
        for j in range(ROI_count):
            file_path = "Curve_Test/QTGUI_Label_Test"+str_t+"/" + "img_result" + "_region_" + str(j) + "_" + str_i + ".jpg"
            img = cv.imread(file_path)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            dest = cv.Sobel(gray, cv.CV_16U, 1, 1)
            mean_index = np.mean(dest)
            tmp = 1 / mean_index
            goal_sum += ROI_weight[j] * tmp
        goal_sum_array.append(goal_sum)
    min_goal_sum=np.min(goal_sum_array)
    mean_goal_sum=np.mean(goal_sum_array)
    max_goal_sum=np.max(goal_sum_array)
    var_goal_sum = np.var(goal_sum_array)
    std_goal_sum = np.std(goal_sum_array)
    min_array.append(min_goal_sum)
    mean_array.append(mean_goal_sum)
    max_array.append(max_goal_sum)
    var_array.append(var_goal_sum)
    std_array.append(std_goal_sum)

print (len(min_array))
plt.plot(min_array)
plt.plot(mean_array)
plt.plot(max_array)
plt.plot(var_array)
plt.plot(std_array)
plt.show()
