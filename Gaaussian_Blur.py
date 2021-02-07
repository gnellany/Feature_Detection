import cv2
import numpy as np

image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blur = cv2.GaussianBlur(gray, (5,5),0)

cv2.imshow('Window Box Name',blur)
cv2.waitKey(0)

print(image)