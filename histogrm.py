import cv2
from matplotlib import pyplot as plt

# read image
img = cv2.imread('FLIR_03860.jpeg')

# convert the image into grayscale before doing histogram equalization
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# image equalization
#equalize_img = cv2.equalizeHist(gray_img)

# creat clahe image
clahe1 = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
#clahe2 = cv2.createCLAHE(clipLimit=40.0,tileGridSize=(8, 8))
#clahe3 = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(64,64 ))
#clahe4 = cv2.createCLAHE(clipLimit=40.0,tileGridSize=(3, 3))

clahe_img1 = clahe1.apply(gray_img)
#clahe_img2 = clahe2.apply(gray_img)
#clahe_img3 = clahe3.apply(gray_img)
#clahe_img4 = clahe4.apply(gray_img)

gblur1 = cv2.GaussianBlur(clahe_img1, (3, 3), 0,0 )
#gblur2 = cv2.GaussianBlur(clahe_img1, (3, 3), 1.1,1.1 )
# show image
#cv2.imshow('ori',img)
#cv2.imshow("gray", gray_img)
#cv2.imshow("equal_image", equalize_img)
cv2.imshow("clahe_image1", clahe_img1)
cv2.imshow("clahe_blur1", gblur1)
#cv2.imshow("clahe_blur2", gblur2)
#cv2.imshow("clahe_image2", clahe_img2)
#cv2.imshow("clahe_image3", clahe_img3)
#cv2.imshow("clahe_image4", clahe_img4)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
# plot image histogram
plt.hist(gray_img.ravel(), 256, [0, 255],label= 'original image')
#plt.hist(equalize_img.ravel(), 256, [0, 255],label= 'equalize image')
plt.hist(clahe_img1.ravel(), 256, [0, 255],label= 'clahe image')
plt.legend()
#plt.grid(True)
plt.show()

'''

