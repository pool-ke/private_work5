# -*- coding: utf-8 -*-

# =============================================================================
# 勾画缺陷轮廓，生成训练集
# =============================================================================

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as s

import cv2

global img
global point1, point2
global flag
global lines

# =============================================================================
# 计算直线交叉点坐标
# =============================================================================

def cross_point(line1,line2):#计算交点函数
    x1=line1[0][0]#取四点坐标
    y1=line1[0][1]
    x2=line1[1][0]
    y2=line1[1][1]
    
    x3=line2[0][0]
    y3=line2[0][1]
    x4=line2[1][0]
    y4=line2[1][1]
    
    k1=(y2-y1)*1.0/(x2-x1)#计算k1,由于点均为整数，需要进行浮点数转化
    b1=y1*1.0-x1*k1*1.0#整型转浮点型是关键
    if (x4-x3)==0:#L2直线斜率不存在操作
        k2=None
        b2=0
    else:
        k2=(y4-y3)*1.0/(x4-x3)#斜率存在操作
        b2=y3*1.0-x3*k2*1.0
    if k2==None:
        x=x3
    else:
        x=(b2-b1)*1.0/(k1-k2)
    y=k1*x*1.0+b1*1.0
    return [int(x),int(y)]

# =============================================================================
# 鼠标动作
# =============================================================================

def on_mouse(event, x, y, flags, param):
    global img, point1, point2, lines
    global flag
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
        point1 = (x,y)
        cv2.circle(img2, point1, 10, (0,255,0), 2)
#        contour.append(point1)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON & flag):      #按住左键拖曳
#        print(contour)
        cv2.line(img2, point1, (x,y), (255,0,0), 2)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:         #左键释放
        point2 = (x,y)
#        if(contour[0][0] == point2[0] and contour[0][1] == point2[1]):
#            flag = False
        cv2.line(img2, point1, point2, (0,0,255), 2) 
#        cv2.imshow('image', img2)
        img = img2
        lines.append([point1, point2])
#        print(lines)
    elif event == cv2.EVENT_RBUTTONDOWN:         #左键释放
        flag = False
    #return min_x, min_y, width, height
    
# =============================================================================
# 获取缺陷轮廓，左键点击勾画，勾画完毕点击右键确认
# =============================================================================
def get_detect_image(f):
    global img
    global flag
    global lines
    global point1
    flag = True
    img = cv2.imread(f)
    img2 = cv2.imread(f)
    cv2.namedWindow('image')
    contour = []
    lines = []
    while(flag):
        cv2.setMouseCallback('image', on_mouse)
        cv2.imshow('image', img)
        cv2.waitKey(0)
        
    contour = []
    if(flag == False):
        i = 0
        contour.append([lines[0][0][0], lines[0][0][1]])
        while(i < len(lines)-1):
            point = cross_point(lines[i], lines[i+1])#计算直线交叉点
            contour.append(point)#记录交叉点坐标
            i += 1
#        contour.append([lines[0][0][0], lines[0][0][1]])
#        print(contour)
    contour = np.array(contour, dtype = np.int32)
#    print(contour)
    contours = np.zeros(img2.shape)
#    cv2.fillConvexPoly(img2, contour, 1)
    cv2.fillConvexPoly(contours, contour, 1)#填充轮廓
    cv2.imshow('image', contours)#显示轮廓
    cv2.waitKey(0)
    return contours

# =============================================================================
# 判断区域是否和网格重叠
# =============================================================================
def if_cross(contours, box):
    #计算重叠区域面积
    box_image = np.zeros(contours.shape)
    x11, y11, x12, y12 = box 
    temp_box = np.array([[x11, y11],[x11, y12],[x12, y11], [x12, y12]], dtype = np.int32)
    cv2.fillConvexPoly(box_image, temp_box, 1)
    #转灰度
    box_image = np.sum(box_image, axis=2)
    contours = np.sum(contours, axis=2)
    is_mat = box_image + contours
    cv2.waitKey(0)
    a = is_mat[is_mat > 1]
    
    re = a.size/((y12-y11)*(x12-x11))
    
    return re
    

import shutil
if __name__ == '__main__':
    train_data = []
    train_label = []
    import os
    base_dir = "./train/"
    files = os.listdir(base_dir)
    for f in files:
        contour = get_detect_image(base_dir + f)
        img1 = cv2.imread(base_dir + f)
    # =============================================================================
    #     切分图片
    # =============================================================================
        patch_y = 16
        patch_x = 16
        
        col = img.shape[1]
        row = img.shape[0]
    
    
        min_x = min(point1[0], point2[0])     
        min_y = min(point1[1], point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] - point2[1])
        box_target = (min_x, min_y, min_x+width, min_y+height)
        
        i = 0
        while(i * patch_x < col):
            j = 0
            while(j * patch_y < row):
                x = i * patch_x
                y = j * patch_y
                box = (x, y, x+patch_x, y+patch_y)
#                coincide = solve_coincide(box_target, box)
                
                re = if_cross(contour, box)
                
                if(re > 0.1):
                    train_label.append(1)
                    cv2.imwrite(base_dir + '/pos/pos' + str(i) + str(j) + f, img1[y:y+patch_y, x:x+patch_x])
                    cv2.rectangle(img, (x, y), (x+patch_x, y+patch_y), (255,0,0), 2) 
                    
                else:
                    train_label.append(-1)
                    cv2.imwrite(base_dir + '/neg/neg' + str(i) + str(j) + f, img1[y:y+patch_y, x:x+patch_x])
#                    cv2.rectangle(img, (x, y), (x+patch_x, y+patch_y), (0,0,255), 2)
                    
                train_data.append(img[y:y+patch_y, x:x+patch_x])
                j += 1
            i += 1
    
        cv2.imshow('image', img)
        cv2.waitKey(0)
        shutil.move(base_dir + f, base_dir + 'finished')#移动文件
        

