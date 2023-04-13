import cv2
import os
import numpy as np

path = "/home/benson/Desktop/fake_mango/mango_pic/file"

for i in range(1,91):
	image = cv2.imread(path +str(i) + ".jpg")
	#cv2.imshow("before", image)
	image2 = cv2.flip(image, 1) # 水平翻轉

	(h, w, d) = image2.shape

	#print(image2.shape)

	center = (w // 2, h // 2)  
	M = cv2.getRotationMatrix2D(center, 20, 1.0)#旋轉 -為順

	rotate_img = cv2.warpAffine(image2, M, (w, h)) 

	#cv2.imshow("Result", rotate_img)
	
	Img_save = "/home/benson/Desktop/redball_test/rotation/" + str(i+180)
	cv2.imwrite(Img_save + '.jpeg',rotate_img)	
	#cv2.imshow("after", image2)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

