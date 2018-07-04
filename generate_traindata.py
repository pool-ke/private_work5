# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt
import numpy as np
import skimage.io as s

import cv2

global img
global point1, point2
def on_mouse(event, x, y, flags, param):
    global img, point1, point2
    img2 = img.copy()
    if event == cv2.EVENT_LBUTTONDOWN:         #左键点击
        point1 = (x,y)
        cv2.circle(img2, point1, 10, (0,255,0), 2)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):      #按住左键拖曳
        cv2.rectangle(img2, point1, (x,y), (255,0,0), 2)
        cv2.imshow('image', img2)
    elif event == cv2.EVENT_LBUTTONUP:         #左键释放
        point2 = (x,y)
        cv2.rectangle(img2, point1, point2, (0,0,255), 2) 
        cv2.imshow('image', img2)
        min_x = min(point1[0], point2[0])     
        min_y = min(point1[1], point2[1])
        width = abs(point1[0] - point2[0])
        height = abs(point1[1] -point2[1])
        cut_img = img[min_y:min_y+height, min_x:min_x+width]
        cv2.imwrite('./temple/orignal.jpg', cut_img)
    #return min_x, min_y, width, height

def get_detect_image(f):
    global img
    img = cv2.imread(f)

    cv2.namedWindow('image')
    cv2.setMouseCallback('image', on_mouse)
    cv2.imshow('image', img)
    cv2.waitKey(0)

def mat_inter(box1,box2):  
    # 判断两个矩形是否相交  
    # box=(xA,yA,xB,yB)  
    x01, y01, x02, y02 = box1  
    x11, y11, x12, y12 = box2  
  
    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)  
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)  
    sax = abs(x01 - x02)  
    sbx = abs(x11 - x12)  
    say = abs(y01 - y02)  
    sby = abs(y11 - y12)  
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:  
        return True  
    else:  
        return False  
  
def solve_coincide(box1,box2):  
    # box=(xA,yA,xB,yB)  
    # 计算两个矩形框的重合度  
    if mat_inter(box1,box2)==True:  
        x01, y01, x02, y02 = box1  
        x11, y11, x12, y12 = box2  
        col=min(x02,x12)-max(x01,x11)  
        row=min(y02,y12)-max(y01,y11)  
        intersection=col*row  
        area1=(x02-x01)*(y02-y01)  
        area2=(x12-x11)*(y12-y11)  
        coincide=intersection/(min(area1,area2))  
        return coincide  
    else:  
        return False 


if __name__ == '__main__':
    train_data = []
    train_label = []
    import os
    base_dir = "./test1/"
    files = os.listdir(base_dir)
    for f in files:
        get_detect_image(base_dir + f)
        img1 = cv2.imread(base_dir + f)
    # =============================================================================
    #     切分图片
    # =============================================================================
        patch_y = 28
        patch_x = 28
        
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
                coincide = solve_coincide(box_target, box)
                if(coincide > 0.4):
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
    
        cv2.imshow('image1', img)
        cv2.waitKey(0)
        

