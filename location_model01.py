#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 09:13:15 2017

@author: root
"""

import sys
import cv2 as cv
import numpy as np
import scipy as sp
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5 import QtCore
from PyQt5.QtWidgets import QFileDialog,QMessageBox,QSlider
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtGui import QPalette
from matplotlib.figure import Figure
import skimage.morphology as sm 
from skimage import filters
import skimage
import os
import copy
import json
import math

class myLabel(QtWidgets.QLabel):
    def __init__(self,parent=None):
        super(myLabel,self).__init__(parent)
        
    def mousePressEvent(self,QMouseEvent):
        if QMouseEvent.button()==Qt.LeftButton:
            image_process.drawing=True
            pointT=QMouseEvent.pos()
            image_process.ROI_X=pointT.x()
            image_process.ROI_Y=pointT.y()
        elif QMouseEvent.button()==Qt.RightButton:
            image_process.mouseprocess5()
#        image_process.mouseprocess()
    
    def mouseReleaseEvent(self,QMouseEvent):
        if QMouseEvent.button()==Qt.LeftButton:
            image_process.drawing=False
            pointT=QMouseEvent.pos()
            image_process.ROI_X_1=pointT.x()
            image_process.ROI_Y_1=pointT.y()
            image_process.mouseprocess2()
        
    def mouseMoveEvent(self,QMouseEvent):
        pointT=QMouseEvent.pos()
        image_process.ROI_X_2=pointT.x()
        image_process.ROI_Y_2=pointT.y()
        image_process.mouseprocess3()
        
    def wheelEvent(self,QMouseWheel):
        delta=QMouseWheel.angleDelta()
        print (delta.y())
        
class myLabel2(QtWidgets.QLabel):
    def __init__(self,parent=None):
        super(myLabel2,self).__init__(parent)
    def mousePressEvent(self,QMouseEvent):
        poinT=QMouseEvent.pos()
        image_process.ROI2_X=poinT.x()
        image_process.ROI2_Y=poinT.y()
        image_process.mouseprocess4()
class Image_Process(QtWidgets.QWidget):
    def __init__(self):
        super(Image_Process,self).__init__()
        self.setGeometry(100,100,1300,900)
        self.setWindowTitle("Image_label")
        self.initUI()
        self.imgOri=None
        self.imgPro1=None
        self.imgPro2=None
        self.imgPro3=None
        self.imgOriIndex=None
        self.imgLabel=None
        self.imgLabel1=None
        self.model=None
        self.modelIndex=None
        self.img_contrast=None
        self.img_model=None
        self.img_contrastIndex=None
        self.img_processed=None
        self.img_currentfilename=None
        self.img_targetIndex=None
        self.img_targetIndex2=None
        self.img_targetIndex3=None
        self.ROI_X=0
        self.ROI_Y=0
        self.ROI_X_1=0
        self.ROI_Y_1=0
        self.ROI_X_2=0
        self.ROI_Y_2=0
        self.ROI2_X=0
        self.ROI2_Y=0
        self.ROI_H=0
        self.ROI_W=0
        self.drawing=False
        self.valuearray=[]
        self.datadict={}
        self.index=1
        self.ratio=0
        self.size=10
        self.searchsize=50
        self.filecount1=0
        self.filecount2=0
        self.abnormalfile1=[]
        self.abnormalfile2=[]
        self.methodname={}
        self.goodnumber=[]
    def initUI(self):
        self.label1=QtWidgets.QLabel(u'File_Path:',self)
        self.label1.move(10,20)
        self.editR=QtWidgets.QLineEdit(self)
        self.editR.move(70,18)
        self.editR.resize(300,18)
        
        self.buttonChoose=QtWidgets.QPushButton(u"ChooseFile",self)
        self.buttonChoose.move(375,18)
        self.buttonChoose.clicked.connect(self.choosefile)
        
        self.buttonLoad1=QtWidgets.QPushButton(u"Loadfiles",self)
        self.buttonLoad1.move(470,18)
        self.buttonLoad1.clicked.connect(self.loadfile1)
        
        self.buttonNext=QtWidgets.QPushButton(u"Next",self)
        self.buttonNext.move(565,18)
        self.buttonNext.clicked.connect(self.nextitem)
        
        self.buttonOK=QtWidgets.QPushButton(u"SaveTotxt",self)
        self.buttonOK.move(820,18)
        self.buttonOK.clicked.connect(self.Savetotxt)
        
        self.buttonROI=QtWidgets.QPushButton(u"OK",self)
        self.buttonROI.move(915,18)
        self.buttonROI.clicked.connect(self.img_process01)
        
        self.label3=QtWidgets.QLabel(u'Height:',self)
        self.label3.move(1010,70)
        self.editH=QtWidgets.QLineEdit(self)
        self.editH.move(1060,70)
        self.editH.resize(50,18)
        self.editH.setText("500")
        
        
        self.label4=QtWidgets.QLabel(u'Width:',self)
        self.label4.move(1010,100)
        self.editW=QtWidgets.QLineEdit(self)
        self.editW.move(1060,100)
        self.editW.resize(50,18)
        self.editW.setText("500")
        
        self.label5=QtWidgets.QLabel(u'PositonX:',self)
        self.label5.move(880,70)
        self.editPositionX=QtWidgets.QLineEdit(self)
        self.editPositionX.move(940,70)
        self.editPositionX.resize(50,18)
        
        self.label6=QtWidgets.QLabel(u'PositonY:',self)
        self.label6.move(880,100)
        self.editPositionY=QtWidgets.QLineEdit(self)
        self.editPositionY.move(940,100)
        self.editPositionY.resize(50,18)
        
        self.label7=QtWidgets.QLabel(u'Filename:',self)
        self.label7.move(660,18)
        self.editT=QtWidgets.QLineEdit(self)
        self.editT.move(720,18)
        self.editT.resize(100,18)
        
        self.label8=QtWidgets.QLabel(u'Size ROI:',self)
        self.label8.move(880,130)
        self.editROIsize=QtWidgets.QLineEdit(self)
        self.editROIsize.move(940,130)
        self.editROIsize.resize(50,18)
        self.editROIsize.setText("10")
        
        self.label9=QtWidgets.QLabel(u'Size Search:',self)
        self.label9.move(1010,130)
        self.editSearchsize=QtWidgets.QLineEdit(self)
        self.editSearchsize.move(1090,130)
        self.editSearchsize.resize(50,18)
        self.editSearchsize.setText("100")
        
        self.allFiles=QtWidgets.QListWidget(self)
        self.allFiles.move(30,40)
        self.allFiles.resize(120,350)
        
        self.allFiles2=QtWidgets.QListWidget(self)
        self.allFiles2.move(30,400)
        self.allFiles2.resize(120,350)
        
        self.labelImg1=myLabel(self)
        self.labelImg1.setAlignment(Qt.AlignTop)
        pe=QPalette()
        self.labelImg1.setAutoFillBackground(True)
        pe.setColor(QPalette.Window,Qt.gray)
        self.labelImg1.setPalette(pe)
        self.labelImg1.move(200,70)
        self.labelImg1.resize(500,500)
        
        self.labelImg2=myLabel2(self)
        self.labelImg2.setAlignment(Qt.AlignTop)
        self.labelImg2.setAutoFillBackground(True)
        pe.setColor(QPalette.Window,Qt.gray)
        self.labelImg2.setPalette(pe)
        self.labelImg2.move(750,170)
        self.labelImg2.resize(300,300)
        
        self.labelImg3=QtWidgets.QLabel(self)
        self.labelImg3.setAlignment(Qt.AlignTop)
        self.labelImg3.setAutoFillBackground(True)
        pe.setColor(QPalette.Window,Qt.gray)
        self.labelImg3.setPalette(pe)
        self.labelImg3.move(750,480)
        self.labelImg3.resize(300,300)
        
        self.allFiles.itemClicked.connect(self.itemClick)
        self.allFiles2.itemClicked.connect(self.itemClick2)
        
#    def mousePressEvent(self,QMouseEvent):
#        pointT=QMouseEvent.pos()
#        print (pointT.x())
#        print (pointT.y())
    def mouseprocess(self):
        print (self.ROI_X)
        print (self.ROI_Y)
        self.editPositionX.setText("%d"%self.ROI_X)
        self.editPositionY.setText("%d"%self.ROI_Y)
        self.ROI_H=self.editH.text()
        self.ROI_W=self.editW.text()
        print (int(self.ROI_X/self.ratio))
        print (int(self.ROI_Y/self.ratio))
        img3=self.imgOri
        cv.imwrite("QTGUI_Label/process003.png",img3)
        self.imgLabel=copy.copy(self.imgOri[:,:])
        cv.rectangle(self.imgLabel,(int(self.ROI_X/self.ratio),int(self.ROI_Y/self.ratio)),(int(self.ROI_X/self.ratio)+int(self.ROI_W),int(self.ROI_Y/self.ratio)+int(self.ROI_H)),(255,0,0),5)
        img2=self.picture_resize(imgOri=self.imgLabel,Label=self.labelImg1)
        cv.imwrite("QTGUI_Label/process001.png",img2)
        qImgT1=QtGui.QPixmap("QTGUI_Label/process001.png")
        self.labelImg1.setPixmap(qImgT1)
        
        self.img_processed=copy.copy(self.imgOri[int(self.ROI_Y/self.ratio):int(self.ROI_Y/self.ratio)+int(self.ROI_H),int(self.ROI_X/self.ratio):int(self.ROI_X/self.ratio)+int(self.ROI_W)])
        img2=self.picture_resize2(imgOri=self.img_processed,Label=self.labelImg2)
        cv.imwrite("QTGUI_Label/process002.png",img2)
        qImgT2=QtGui.QPixmap("QTGUI_Label/process002.png")
        self.labelImg2.setPixmap(qImgT2)
#        self.img_processed=self.imgOri[int(self.ROI_Y/self.ratio):int(self.ROI_Y/self.ratio)+int(self.ROI_H)),int(self.ROI_X/self.ratio):(int(self.ROI_X/self.ratio)+int(self.ROI_W))]
#        self.img_processed=self.imgOri[1000:1600,1800:2400]
#        img2=self.picture_resize(imgOri=self.img_processed,Label=self.labelImg2)
#        cv.imwrite("QTGUI_Label/process002.png",img2)
#        qImgT=QtGui.QPixmap("QTGUI_Label/process002.png")
#        self.labelImg2.setPixmap(qImgT)
#        geomertyarray=[self.ROI_X,self.ROI_Y,self.ROI_W,self.ROI_H]
#        self.valuearray.append(geomertyarray)
#        if self.datadict[self.img_currentfilename]==None:
#            self.datadict[self.img_currentfilename]=[]
#        self.datadict[self.img_currentfilename]=self.valuearray
#        self.visualize(self)
        
        
#        self.editH.setText("%d"%self.label_y)

    def mouseprocess2(self):
        print (self.ROI_X_1)
        print (self.ROI_Y_1)
        positionX=int(self.ROI_X/self.ratio)
        positionY=int(self.ROI_Y/self.ratio)
        self.searchsize=int(self.editSearchsize.text())
        self.editPositionX.setText("%d"%positionX)
        self.editPositionY.setText("%d"%positionY)
        self.ROI_W=int(self.ROI_X_1/self.ratio)-int(self.ROI_X/self.ratio)
        self.ROI_H=int(self.ROI_Y_1/self.ratio)-int(self.ROI_Y/self.ratio)
        self.editH.setText(str(self.ROI_H))
        self.editW.setText(str(self.ROI_W))
        
        self.imgLabel=copy.copy(self.imgOri[:,:])
        cv.imwrite("QTGUI_Label/process003.png",self.imgOri)
#        cv.line(self.imgLabel,(int(self.ROI_X/self.ratio),int(self.ROI_Y/self.ratio)),(int(self.ROI_X_1/self.ratio),int(self.ROI_Y_1/self.ratio)),(255,0,0),5)
        cv.rectangle(self.imgLabel,(int(self.ROI_X/self.ratio),int(self.ROI_Y/self.ratio)),(int(self.ROI_X_1/self.ratio),int(self.ROI_Y_1/self.ratio)),(255,0,0),5)
        img2=self.picture_resize(imgOri=self.imgLabel,Label=self.labelImg1)
        cv.imwrite("QTGUI_Label/process001.png",img2)
        qImgT1=QtGui.QPixmap("QTGUI_Label/process001.png")
        self.labelImg1.setPixmap(qImgT1)
        
        self.img_processed=copy.copy(self.imgOri[int(self.ROI_Y/self.ratio):int(self.ROI_Y_1/self.ratio),int(self.ROI_X/self.ratio):int(self.ROI_X_1/self.ratio)])
        self.img_model=copy.copy(self.imgOri[int(self.ROI_Y/self.ratio)-int(self.searchsize):int(self.ROI_Y_1/self.ratio)+int(self.searchsize),int(self.ROI_X/self.ratio)-int(self.searchsize):int(self.ROI_X_1/self.ratio)+int(self.searchsize)])
        img2=self.picture_resize2(imgOri=self.img_processed,Label=self.labelImg2)
        cv.imwrite("QTGUI_Label/process005.png",self.img_processed)
        cv.imwrite("QTGUI_Label/process006.png",self.img_model)
        cv.imwrite("QTGUI_Label/process002.png",img2)
        qImgT2=QtGui.QPixmap("QTGUI_Label/process002.png")
        self.labelImg2.setPixmap(qImgT2)
        
    def mouseprocess3(self):
        print (self.ROI_X_2)
        print (self.ROI_Y_2)
        positionX=int(self.ROI_X/self.ratio)
        positionY=int(self.ROI_Y/self.ratio)
        self.editPositionX.setText("%d"%positionX)
        self.editPositionY.setText("%d"%positionY)
        self.ROI_W=int(self.ROI_X_2/self.ratio)-int(self.ROI_X/self.ratio)
        self.ROI_H=int(self.ROI_Y_2/self.ratio)-int(self.ROI_Y/self.ratio)
        self.editH.setText(str(self.ROI_H))
        self.editW.setText(str(self.ROI_W))
        
        self.imgLabel1=copy.copy(self.imgOri[:,:])
        cv.rectangle(self.imgLabel1,(int(self.ROI_X/self.ratio),int(self.ROI_Y/self.ratio)),(int(self.ROI_X_2/self.ratio),int(self.ROI_Y_2/self.ratio)),(255,0,0),5)
        img2=self.picture_resize(imgOri=self.imgLabel1,Label=self.labelImg1)
        cv.imwrite("QTGUI_Label/process001.png",img2)
        qImgT1=QtGui.QPixmap("QTGUI_Label/process001.png")
        self.labelImg1.setPixmap(qImgT1)
#        if (self.drawing):
#            print ("111")
#        else:
#            print ("222")
    def mouseprocess4(self):
        print(self.ROI2_X)
        print(self.ROI2_Y)
        
    def mouseprocess5(self):
        cv.rectangle(self.imgLabel,(int(self.ROI_X/self.ratio),int(self.ROI_Y/self.ratio)),(int(self.ROI_X_1/self.ratio),int(self.ROI_Y_1/self.ratio)),(0,0,255),5)
        img2=self.picture_resize(imgOri=self.imgLabel,Label=self.labelImg1)
        cv.imwrite("QTGUI_Label/process001.png",img2)
        qImgT1=QtGui.QPixmap("QTGUI_Label/process001.png")
        self.labelImg1.setPixmap(qImgT1)
    def choosefile(self):
        directory1=QFileDialog.getExistingDirectory(self,"get directory","/home/huawei")
        self.editR.setText(str(directory1))
    
    def loadfile1(self):
        imgPath=self.editR.text()
        print (imgPath)
        if os.path.isdir(imgPath):
            allImgs=os.listdir(imgPath)
            for imgTemp in allImgs:
                self.allFiles.addItem(imgTemp)
                
        else:
            QMessageBox.information(self,"Warning","The Diretory is Not Exist!",QMessageBox.Ok)
            
        allImgs=os.listdir(imgPath)
        for i in range(len(allImgs)):
            temp=imgPath+"/"+allImgs[i]
            temp1="QTGUISample/"+allImgs[i]
            print (temp)
            img1=cv.imread(temp)
            cv.imwrite(temp1,img1)
            
    def itemClick(self):
        temp=self.editR.text()+"/"+self.allFiles.currentItem().text()
        self.img_currentfilename=self.allFiles.currentItem().text()
        self.valuearray=[]
        self.imgOri=cv.imread(str(temp))
        print (self.imgOri)
        print (self.imgOri.shape[0])
        print (self.imgOri.shape[1])
        img2=self.picture_resize(imgOri=self.imgOri,Label=self.labelImg1)
        cv.imwrite("QTGUI_Label/process001.png",img2)
        qImgT=QtGui.QPixmap("QTGUI_Label/process001.png")
        self.labelImg1.setPixmap(qImgT)
        self.index=1
     
    def itemClick2(self):
        file_path1="QTGUIPROCESS/"
        file_path2="QTGUITarget/"
        file_path3="QTGUITarget13/"
        temp1=file_path1+self.allFiles2.currentItem().text()
        self.imgPro1=cv.imread(str(temp1))
        img2=self.picture_resize(imgOri=self.imgPro1,Label=self.labelImg1)
        cv.imwrite("QTGUI_Label/process001.png",img2)
        qImgT=QtGui.QPixmap("QTGUI_Label/process001.png")
        self.labelImg1.setPixmap(qImgT)
        temp2=file_path2+self.allFiles2.currentItem().text()
        self.imgPro2=cv.imread(str(temp2))
        img2=self.picture_resize2(imgOri=self.imgPro2,Label=self.labelImg2)
        cv.imwrite("QTGUI_Label/process002.png",img2)
        qImgT=QtGui.QPixmap("QTGUI_Label/process002.png")
        self.labelImg2.setPixmap(qImgT)
        temp3=file_path3+self.allFiles2.currentItem().text()
        self.imgPro3=cv.imread(str(temp3))
        img3=self.picture_resize2(imgOri=self.imgPro3,Label=self.labelImg3)
        cv.imwrite("QTGUI_Label/process008.png",img3)
        qImgT=QtGui.QPixmap("QTGUI_Label/process008.png")
        self.labelImg3.setPixmap(qImgT)
        
        return 0
    def nextitem(self):
        currentrow=self.allFiles.currentRow()
        print (currentrow)
        if currentrow==self.allFiles.count()-1:
            QMessageBox.information(self,"It is the last item!",QMessageBox.Ok)
        else:
            self.allFiles.setCurrentRow(currentrow+1)
            self.itemClick()
        
    def Savetotxt(self):
        file_path_save=self.editT.text()
        if file_path_save.split('.')[-1]=="txt":
            print (123)
            output=open(file_path_save,'w')
            output.write(str(json.dumps(self.datadict,indent=1)))
            output.close()
        else:
            print (321)
            QMessageBox.information(self,"It is not the txt file!",QMessageBox.Ok)
            
    def picture_resize(self,imgOri,Label):
        height=imgOri.shape[0]
        width=imgOri.shape[1]
        
        if (height>width):
            ratioY=Label.height()/(height+0.0)
            self.ratio=ratioY
            print ("Y:"+str(self.ratio))
            height2=Label.height()
            width2=int(width*ratioY+0.5)
            imgPro=cv.resize(imgOri,(width2,height2))
        else:
            ratioX=Label.width()/(width+0.0)
            self.ratio=ratioX
            print ("X:"+str(self.ratio))
            width2=Label.width()
            height2=int(height*ratioX+0.5)
            imgPro=cv.resize(imgOri,(width2,height2))
        
        return imgPro
    
    def picture_resize2(self,imgOri,Label):
        height=imgOri.shape[0]
        width=imgOri.shape[1]
        
        if (height>width):
            ratioY=Label.height()/(height+0.0)
            print ("Y:"+str(self.ratio))
            height2=Label.height()
            width2=int(width*ratioY+0.5)
            imgPro=cv.resize(imgOri,(width2,height2))
        else:
            ratioX=Label.width()/(width+0.0)
            print ("X:"+str(self.ratio))
            width2=Label.width()
            height2=int(height*ratioX+0.5)
            imgPro=cv.resize(imgOri,(width2,height2))
        
        return imgPro
    
    def img_process01(self):
        print (self.allFiles.count())
        self.size=int(self.editROIsize.text())
        self.searchsize=int(self.editSearchsize.text())
        print (self.size)
        print(self.allFiles.count())
        datadict={}
#        for i in range(self.allFiles.count()):
        for i in range(self.allFiles.count()):
            temp="QTGUISample/"+self.allFiles.item(i).text()
            temp1="QTGUIPROCESS/"+self.allFiles.item(i).text()
            temp3="QTGUITarget/"+self.allFiles.item(i).text()
            temp4="QTGUITarget3/"+self.allFiles.item(i).text()
            temp5="QTGUITarget5/"+self.allFiles.item(i).text()
            temp6="QTGUITarget7/"+self.allFiles.item(i).text()
            temp7="QTGUITarget13/"+self.allFiles.item(i).text()
            temp2=self.allFiles.item(i).text()
            self.allFiles2.addItem(temp2)
            print (temp)
            print (temp1)
            self.img_processed=cv.imread("QTGUI_Label/process005.png",0)
            self.imgOriIndex=cv.imread(str(temp),0)
#            img_temp1=copy.copy(self.imgOriIndex[:,:])
#            cv.rectangle(img_temp1,(int(self.ROI_X/self.ratio)-self.size,int(self.ROI_Y/self.ratio)-self.size),(int(self.ROI_X/self.ratio)+self.ROI_W+2*self.size,int(self.ROI_Y/self.ratio)+self.ROI_H+2*self.size),(0,0,255),5)
#            cv.imwrite(temp1,img_temp1)
            self.img_targetIndex=copy.copy(self.imgOriIndex[int(self.ROI_Y/self.ratio)-self.searchsize:int(self.ROI_Y_1/self.ratio)+self.searchsize,int(self.ROI_X/self.ratio)-self.searchsize:int(self.ROI_X_1/self.ratio)+self.searchsize])
#            self.img_targetIndex=copy.copy(self.imgOriIndex)
            cv.imwrite(temp3,self.img_targetIndex)
            self.img_processed=cv.equalizeHist(self.img_processed)
            self.img_targetIndex=cv.equalizeHist(self.img_targetIndex)
            print (self.img_processed.shape)
            print (self.img_targetIndex.shape)
#            gray1=cv.cvtColor(self.img_processed,cv.COLOR_BGR2GRAY)
#            gray2=cv.cvtColor(self.img_targetIndex,cv.COLOR_BGR2GRAY)
            gray1=self.img_processed
            gray2=self.img_targetIndex
            h1,w1=gray1.shape[:2]
            h2,w2=gray2.shape[:2]
            sift=cv.xfeatures2d.SIFT_create()
            (kp1,des1)=sift.detectAndCompute(gray1,None)
            (kp2,des2)=sift.detectAndCompute(gray2,None)
            print ("#kp1:{},descriptors1:{}".format(len(kp1),des1.shape))
#            print ("#kp2:{},descriptors2:{}".format(len(kp2),des2.shape))
            if (len(kp2)>1):
                FLANN_INDEX_KDTREE=0
                index_params=dict(algorithm=FLANN_INDEX_KDTREE,trees=3)
                search_params=dict(checks=50)
                flann=cv.FlannBasedMatcher(index_params,search_params)
                print (des2.shape)
                matches=flann.knnMatch(des1,des2,k=2)
    
                print ('matches..:',len(matches))
                good=[]
                for m,n in matches:
                    if m.distance<0.5*n.distance:
                        good.append(m)
                print ("good:",len(good))
                self.goodnumber.append(len(good))
                if (len(good)>0):
                    array_x=[]
                    array_y=[]
                    view=sp.zeros((max(h1,h2),w1+w2,3),sp.uint8)
                    view[:h1,:w1,0]=gray1
                    view[:h2,w1:,0]=gray2
                    view[:,:,1]=view[:,:,0]
                    view[:,:,2]=view[:,:,0]
                    for m in good:
    #                print (int(kp1[m.queryIdx].pt[0]),int(kp1[m.queryIdx].pt[1]))
    #                print (int(kp2[m.trainIdx].pt[0]),int(kp2[m.trainIdx].pt[1]))
    #                print (int(self.ROI_X/self.ratio)-self.searchsize+int(kp2[m.trainIdx].pt[0])-int(kp1[m.queryIdx].pt[0]))
    #                print (int(self.ROI_Y/self.ratio)-self.searchsize+int(kp2[m.trainIdx].pt[1])-int(kp1[m.queryIdx].pt[1]))
                        cv.line(view,(int(kp1[m.queryIdx].pt[0]),int(kp1[m.queryIdx].pt[1])),(int(kp2[m.trainIdx].pt[0]+w1),int(kp2[m.trainIdx].pt[1])),(255,0,0))
                        temp_x=int(self.ROI_X/self.ratio)-self.searchsize+int(kp2[m.trainIdx].pt[0])-int(kp1[m.queryIdx].pt[0])
                        temp_y=int(self.ROI_Y/self.ratio)-self.searchsize+int(kp2[m.trainIdx].pt[1])-int(kp1[m.queryIdx].pt[1])
                        array_x.append(temp_x)
                        array_y.append(temp_y)
                    position_x=int(np.mean(array_x))
                    position_y=int(np.mean(array_y))
                    print (position_x)
                    print (position_y)
                    img_temp1=copy.copy(self.imgOriIndex[:,:])
                    cv.rectangle(img_temp1,(position_x,position_y),(position_x+self.ROI_W,position_y+self.ROI_H),(0,0,255),5)
                    cv.imwrite(temp1,img_temp1)
#                    img_target_temp=copy.copy(self.imgOriIndex[position_y-100:position_y+self.ROI_H+100,position_x-100:position_x+self.ROI_W+100])
                    info_temp=[]
                    info_temp.append(position_x)
                    info_temp.append(position_y)
                    info_temp.append(int(self.ROI_W))
                    info_temp.append(int(self.ROI_H))
                    datadict[temp2]=info_temp                    
                    img_target_temp=copy.copy(self.imgOriIndex[position_y:position_y+self.ROI_H,position_x:position_x+self.ROI_W])
                    img_target_temp_2=copy.copy(self.imgOriIndex[position_y-self.searchsize:position_y+self.ROI_H+self.searchsize,position_x-self.searchsize:position_x+self.ROI_W+self.searchsize])
                    cv.imwrite(temp3,img_target_temp)
                    cv.imwrite(temp5,img_target_temp_2)
                    methods=['cv.TM_CCOEFF','cv.TM_CCOEFF_NORMED','cv.TM_CCORR','cv.TM_CCORR_NORMED','cv.TM_SQDIFF','cv.TM_SQDIFF_NORMED']
                    meth=methods[0]
                    method=eval(meth)
                    img_template=cv.imread("QTGUI_Label/process005.png",0)
                    h,w=img_template.shape[:2]
                    img_search=cv.imread(temp5,0)
                    angle_best=0
                    max_value=0
                    for Angle in np.arange(-3,3,0.1): 
                        (H_temp)=cv.getRotationMatrix2D(center=(img_search.shape[1]/2,img_search.shape[0]/2),angle=Angle,scale=1)
                        result=cv.warpAffine(img_search,H_temp,(img_search.shape[1],img_search.shape[0]))
                        res=cv.matchTemplate(result,img_template,method)
                        min_val,max_val,min_loc,max_loc=cv.minMaxLoc(res)
                        if max_val>max_value:
                            max_value=max_val
                            angle_best=Angle
                    print (angle_best)
                    (H)=cv.getRotationMatrix2D(center=(img_search.shape[1]/2,img_search.shape[0]/2),angle=angle_best,scale=1)
                    result1=cv.warpAffine(img_search,H,(img_search.shape[1],img_search.shape[0]))
                    res2=cv.matchTemplate(result1,img_template,method)
                    min_val,max_val,min_loc,max_loc=cv.minMaxLoc(res2)
                    if method in [cv.TM_SQDIFF,cv.TM_SQDIFF_NORMED]:
                        top_left=min_loc
                    else:
                        top_left=max_loc
                    result2=copy.copy(result1[top_left[1]:top_left[1]+h,top_left[0]:top_left[0]+w])
                    cv.imwrite(temp7,result2)
#                    self.img_targetIndex2=cv.imread(temp6,0)
#                    self.img_targetIndex3=cv.imread(temp6)
#                    self.img_targetIndex2=cv.equalizeHist(self.img_targetIndex2)
#                    gray1=self.img_processed
#                    gray2=self.img_targetIndex2
#                    sift=cv.xfeatures2d.SIFT_create()
#                    (kp1,des1)=sift.detectAndCompute(gray1,None)
#                    (kp2,des2)=sift.detectAndCompute(gray2,None)
#                    FLANN_INDEX_KDTREE=0
#                    index_params=dict(algorithm=FLANN_INDEX_KDTREE,trees=3)
#                    search_params=dict(checks=50)
#                    flann=cv.FlannBasedMatcher(index_params,search_params)
#                    matches=flann.knnMatch(des1,des2,k=2)
#        
#                    print ('matches..:',len(matches))
#                    good=[]
#                    for m,n in matches:
#                        if m.distance<0.5*n.distance:
#                            good.append(m)
#                    print ("good:",len(good))
#                    array_x=[]
#                    array_y=[]
#                    for m in good:
#                        temp_x=int(kp2[m.trainIdx].pt[0])-int(kp1[m.queryIdx].pt[0])
#                        temp_y=int(kp2[m.trainIdx].pt[1])-int(kp1[m.queryIdx].pt[1])
#                        array_x.append(temp_x)
#                        array_y.append(temp_y)
#                    position_x=int(np.mean(array_x))
#                    position_y=int(np.mean(array_y))
#                    print (position_x)
#                    print (position_y)
#                    img_target_temp3=copy.copy(self.img_targetIndex3[position_y:position_y+self.ROI_H,position_x:position_x+self.ROI_W])
#                    cv.imwrite(temp7,img_target_temp3)
                    cv.imwrite(temp4,view)
                    self.methodname[temp2]="SIFT"
                else:
                    self.methodname[temp2]="No SIFT"
            else:
                self.methodname[temp2]="NO SIFT"
        print ("333")
        print (self.methodname)
        print ("444")
        print (self.goodnumber)
        print ("123")
        print (len(datadict))
        file_path_save="info_loc_T002.txt"
        output=open(file_path_save,'w')
        output.write(str(json.dumps(datadict,indent=1)))
        output.close()
if __name__=="__main__":
    app=QtWidgets.QApplication(sys.argv)
    image_process=Image_Process()
    image_process.show()
    sys.exit(app.exec_())

