#importing libraries to be used
import cv2
import numpy as np
import matplotlib.pyplot as plt
#Using cv2 to read the image
image = cv2.imread('bottles.jpg')
#Using plt to see the image
plt.imshow(image)
#Converting the image to Gray to reduce the noise of the Bg
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#Converting the image to Gray to reduce the noise of the Bg
plt.imshow(gray,cmap="gray")
#Converting the image blur to make it easy for computer vision to identify objects and reduce noise
blur = cv2.GaussianBlur(gray,(11,11),0)
plt.imshow(blur, cmap="gray")
#Here 50,30 means any pixel value which is lowered threshold which means any pixel gradient value whose value is less than 50 just ignore
#Here 150,250 means any pixel value which is upper threshold which means any pixel gradient value whose value is more than 150 just ignore
#Here 3 means kernel size for the sobel operations which helps me in identify edges so smoothly
canny = cv2.Canny(blur, 30,250,3)
plt.imshow(canny, cmap='gray')
#Here Dilation is used to make edges more visible
dilated = cv2.dilate(canny, (1,1), iterations=0)
plt.imshow(dilated, cmap='gray')
#Defining Contours
(cnt, hierarchy) = cv2.findContours(dilated.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (0,255,0),2)
print("Bottles in the Image is: ",len(cnt))
plt.imshow(rgb)