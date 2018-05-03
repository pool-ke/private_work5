2#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:08:03 2017

@author: root
"""

import numpy as np
import json

def IOU(Reframe,GTframe):
    x1=Reframe[0]
    y1=Reframe[1]
    width1=Reframe[2]
    height1=Reframe[3]
    
    x2=GTframe[0]
    y2=GTframe[1]
    width2=GTframe[2]
    height2=GTframe[3]
    
    endx=max(x1+width1,x2+width2)
    startx=min(x1,x2)
    width=width1+width2-(endx-startx)
    
    endy=max(y1+height1,y2+height2)
    starty=min(y1,y2)
    height=height1+height2-(endy-starty)
    
    if width<=0 or height<=0:
        ratio=0
    else:
        Area=width*height
        Area1=width1*height1
        Area2=width2*height2
        ratio=Area*1.0/(Area1+Area2-Area)
    return ratio

def PR(Reframe,GTframe):
    x1=Reframe[0]
    y1=Reframe[1]
    width1=Reframe[2]
    height1=Reframe[3]
    
    x2=GTframe[0]
    y2=GTframe[1]
    width2=GTframe[2]
    height2=GTframe[3]
    
    endx=max(x1+width1,x2+width2)
    startx=min(x1,x2)
    width=width1+width2-(endx-startx)
    
    endy=max(y1+height1,y2+height2)
    starty=min(y1,y2)
    height=height1+height2-(endy-starty)
    
    if width<=0 or height<=0:
        Precision=0
        Recall=0
    else:
        Area=width*height
        Area1=width1*height1
        Area2=width2*height2
        Precision=Area*1.0/Area2
        Recall=Area*1.0/Area1
    return Precision,Recall



file_path1="info_loc002.txt"
file_path2="info_loc_T002.txt"

input1=open(file_path1,'r')
input2=open(file_path2,'r')
load_dict1=json.load(input1)
load_dict2=json.load(input2)
print (len(load_dict1))
print (len(load_dict2))

#for name in load_dict1.keys():
#    print (name)
#    print (load_dict1[name])
#    print (load_dict2[name])
ratio_array=[]
precision_array=[]
recall_array=[]
for name in load_dict1.keys():
    ratio_temp=IOU(load_dict1[name],load_dict2[name])
    precision_temp,recall_temp=PR(load_dict1[name],load_dict2[name])
    ratio_array.append(ratio_temp)
    precision_array.append(precision_temp)
    recall_array.append(recall_temp)
    

ratio_min=np.min(ratio_array)
ratio_max=np.max(ratio_array)
ratio_mean=np.mean(ratio_array)
ratio_var=np.var(ratio_array)
ratio_std=np.std(ratio_array)

precision_min=np.min(precision_array)
precision_max=np.max(precision_array)
precision_mean=np.mean(precision_array)

recall_min=np.min(recall_array)
recall_max=np.max(recall_array)
recall_mean=np.mean(recall_array)

print ("ratio_IOU")
print (ratio_min)
print (ratio_max)
print (ratio_mean)

print ("Precision")
print (precision_min)
print (precision_max)
print (precision_mean)

print ("Recall")
print (recall_min)
print (recall_max)
print (recall_mean)

