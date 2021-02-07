import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_intrest(image):
    height = image.shape[0]
    polygon = np.array([[(200,height),(1100, height),(550,250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygon,255)
    return mask



image = cv2.imread('test_image.jpg')
lane_image = np.copy(image)
canny = canny(lane_image)

plt.imshow(region_of_intrest(canny))
plt.show()

print(image)