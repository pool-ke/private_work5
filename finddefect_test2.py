

import numpy as np
import cv2
import os
from skimage import filters,io, measure, color
import skimage.morphology as sm
import random
from numpy import array


'''for many character 2018/05/22'''


def get_binary(img):
	'''for the reconstruction image'''
	if(len(img.shape)==3):
		imgf=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	else:
		imgf=img
	poly_img=np.zeros(imgf.shape,dtype=np.uint8)
	imgff=cv2.GaussianBlur(imgf,(5,5),0)
	'''get the binary image'''
	_,imag=cv2.threshold(imgff,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	_,contours,hire=cv2.findContours(imag,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	areas=np.zeros(len(contours))
	idx=0
	for cont in contours:
		areas[idx]=cv2.contourArea(cont)
		idx=idx+1
	'''sort the contour by the area size surrounded by the contour'''
	areas_s=cv2.sortIdx(areas,cv2.SORT_DESCENDING|cv2.SORT_EVERY_COLUMN)
	return imag,areas_s,contours,hire

def get_binary3(img,weigtht=0.5):
	'''original version to find the binary image of img'''
	if(len(img.shape)==3):
		imgf=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	else:
		imgf=img
	poly_img=np.zeros(imgf.shape,dtype=np.uint8)
	imgfilter=filters.gaussian(imgf,weigtht)
	thresh=filters.threshold_otsu(imgfilter)
	poly_img=(imgfilter>=thresh)*1.0
	return poly_img

def compareimage(img,img2):
	if(len(img.shape)==3):
		mix=np.zeros([img.shape[0],img.shape[1]*2,3],np.uint8)
		mix[0:img.shape[0],0:img.shape[1],:]=img
		mix[0:img2.shape[0]:,img2.shape[1]:,:]=img2
	else:
		mix=np.zeros([img.shape[0],img.shape[1]*2],np.uint8)
		mix[0:img.shape[0],0:img.shape[1]]=img
		mix[0:img2.shape[0]:,img2.shape[1]:]=img2
	return mix

if(not os.path.exists('./tttttttt')):
    os.mkdir('./tttttttt')

'''whether show the image'''
showimage=False
'''whether save the image'''
saveimage=not showimage
'''the min distance between the point and the contours'''
mindistancein=2
mindistanceout=2
path='/home/huawei/Lanjiulong/CAEtest/2CAE2image/20180521'
id=0

if __name__=='__main__':
	for id in range(32):
		bimg = cv2.imread(path+'/test1/{}.bmp'.format(id))
		bimg2 = cv2.imread(path+'/test1_template/{}.bmp'.format(id))
		img=cv2.resize(bimg,(0,0),fx=2,fy=2,interpolation=cv2.INTER_CUBIC)
		img2=cv2.resize(bimg2,(0,0),fx=2,fy=2,interpolation=cv2.INTER_CUBIC)

		'''img is the original image'''
		poly_img=get_binary3(img)
		'''img2 is the reconstruction image'''
		poly_img2,area_s,contour,hire=get_binary(img2)

		if showimage:
			cv2.imshow("poly_img",poly_img)
			cv2.imshow("poly_img2",poly_img2)
			poly_mix=compareimage(poly_img,poly_img2)
			image_mix=compareimage(img,img2)
			cv2.imshow("image compare",image_mix)
			cv2.imshow("binary compare",poly_mix)

		'''poly_img3 is the add of poly_img2 and poly_img'''
		poly_img3=np.zeros(poly_img.shape,dtype=np.uint8)
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				if(poly_img[i,j]>0 or poly_img2[i,j]>0):
					poly_img3[i,j]=255
		if showimage:
			cv2.imshow("poly_img3",poly_img3)

		'''poly_img4 is the intersection of the poly_img and poly_img2'''
		'''in the contour'''
		poly_img4=np.zeros(poly_img.shape,dtype=np.uint8)
		area=len(area_s)
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):
				for index in range(area):
						if(hire[0][area_s[index]][0][3]==-1 and cv2.pointPolygonTest(contour[int(area_s[index])],(j,i),False)>=0):
							# print(area_s[index],index)
							if(poly_img[i,j]>0 and poly_img2[i,j]>0):
								poly_img4[i,j]=255
		if showimage:
			cv2.imshow("poly_img4",poly_img4)

		# ===============================INNER====================================
		'''inner_defect is the defect surrounded by the coutour'''
		inner_defect=np.abs(poly_img2-poly_img4)
		num_for_open = 2 #step for opening operator
		#inner_defect = sm.opening(inner_defect,sm.square(num_for_open))


		'''filter the point close to the coutour'''
		for i in range(inner_defect.shape[0]):
			for j in range(inner_defect.shape[1]):
				if(inner_defect[i,j]>0):
					min=99
					for index in range(area):
						if(min>abs(cv2.pointPolygonTest(contour[int(area_s[index])],(j,i),True))):
							min=abs(cv2.pointPolygonTest(contour[int(area_s[index])],(j,i),True))
					'''the point close to the contour would be filtered'''
					if(min<mindistancein):
						inner_defect[i,j]=0


		'''convert the defect binary image to a color image'''
		inner_defectcolor=np.zeros([inner_defect.shape[0],inner_defect.shape[1],3],np.uint8)
		inner_defectcolor[:,:,2]=(inner_defect>0)*255
		if showimage:
			cv2.imshow("inner_defectcolor",inner_defectcolor)

		# ===============================OUTER====================================
		'''outer_defect is the defect which is not surrounded by the coutour'''
		outer_defect=np.abs(poly_img2-poly_img3)
		outer_defect=sm.opening(outer_defect,sm.square(num_for_open))
		for i in range(outer_defect.shape[0]):
			for j in range(outer_defect.shape[1]):
				if(outer_defect[i,j]>0):
					min=99
					for index in range(area):
						if(min>abs(cv2.pointPolygonTest(contour[int(area_s[index])],(j,i),True))):
							min=abs(cv2.pointPolygonTest(contour[int(area_s[index])],(j,i),True))
					'''the point close to the contour would be filtered'''
					if(min<mindistanceout):
						outer_defect[i,j]=0

		'''convert the defect binary image to a color image'''
		outer_defectcolor=np.zeros([outer_defect.shape[0],outer_defect.shape[1],3],np.uint8)
		outer_defectcolor[:,:,2]=(outer_defect>0)*255
		if showimage:
			cv2.imshow("outer_defectcolor",outer_defectcolor)

		# ===============================DEFECT====================================
		'''create the defect color image '''
		defectcolor=np.zeros([poly_img.shape[0],poly_img.shape[1],3],dtype=np.uint8)
		defect=inner_defect+outer_defect

		''''''
		defect=sm.opening(defect,sm.square(3))


		defectcolor[:,:,2]=(defect>0)*255

		'''mix the defect with the binary image'''
		mix=cv2.addWeighted(defectcolor,0.5,img,0.5,0)
		#mix=imgg+img_dif2
		if showimage:
			cv2.imshow("mix",mix)

		'''create the color image of the binary image'''
		poly_imgg=np.zeros([mix.shape[0],mix.shape[1],3],np.uint8)
		poly_img2g=np.zeros([mix.shape[0],mix.shape[1],3],np.uint8)
		poly_imgg[:,:,1]=(poly_img>0)*255
		poly_img2g[:,:,1]=(poly_img2>0)*255

		mix2=np.zeros([mix.shape[0],mix.shape[1]*7,3],np.uint8)
		mix2[0:mix.shape[0],0:mix.shape[1]]=img
		mix2[0:mix.shape[0],mix.shape[1]:2*mix.shape[1]]=img2
		mix2[0:mix.shape[0],2*mix.shape[1]:3*mix.shape[1]]=poly_imgg
		mix2[0:mix.shape[0],3*mix.shape[1]:4*mix.shape[1]]=poly_img2g
		mix2[0:mix.shape[0],4*mix.shape[1]:5*mix.shape[1]]=inner_defectcolor
		mix2[0:mix.shape[0],5*mix.shape[1]:6*mix.shape[1]]=outer_defectcolor
		mix2[0:mix.shape[0]:,6*mix.shape[1]:]=mix

		if showimage:
			cv2.imshow("mix2",mix2)
		if saveimage:
			cv2.imwrite('tttttttt/{}.bmp'.format(id),mix2)
			print("save {} bmp".format(id))
		if showimage:
			cv2.waitKey(0)
			cv2.destroyAllWindows()