import cv2
import numpy as np

def region_of_intrest(image):
    height = image.shape[0]
    polygon = np.array([[(100,height),(1550, height),(800,100), (50,600)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygon,255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def harris(image):
    dst = cv2.cornerHarris(image, 2, 3, 0.04)
    fname[dst > 0.01 * dst.max()] = [0, 0, 255]
    return dst

def corners(image):
    corners = cv2.goodFeaturesToTrack((harris(region_of_intrest(canny(fname)))), 25, 0.01, 10)
    corners = np.int0(corners)

    for i in corners:
        x, y = i.ravel()
        cv2.circle(image, (x, y), 3, 255, -1)
    return image


images = cv2.VideoCapture('test6.mp4')

kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
forback = cv2.bgsegm.createBackgroundSubtractorGMG()

while True:
        ret, fname = images.read()

        fgmask = forback.apply(fname)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernal)


        cv2.imshow('DST', corners(fgmask))
        if cv2.waitKey(1) & 0xFF == ord ('q'):
            break

cv2.destroyAllWindows()