import cv2
import numpy as np

def harris(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    image[dst > 0.01 * dst.max()] = [0, 0, 255]
    return image

images = cv2.VideoCapture('test6.mp4')

while True:
        ret, fname = images.read()


        cv2.imshow('Harris Corner Detection', harris(fname))
        if cv2.waitKey(1) & 0xFF == ord ('q'):
            break

cv2.destroyAllWindows()
