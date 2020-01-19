import cv2 as cv
import numpy as np

im = cv.imread('images/backyard.jpeg')

print(im.shape)
height, width, channel = im.shape[:3]
print("Height of the image is: ", height)
print("Width of the image is: ", width)
print("Number of channels in image: ", channel)

b = im[:,:,0]
print(b)
g = im[:,:,1]
r = im[:,:,2]

cv.imshow("Blue Channel", b)
cv.imshow("Green Channel", g)
cv.imshow("Red Channel", r)
cv.imshow("Landscape", im)
cv.waitKey(0)
cv.destroyAllWindows()