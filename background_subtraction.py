import numpy as np
import cv2

def canny(image):
    blur = cv2.GaussianBlur(image, (5, 5), 1)
    canny = cv2.Canny(blur, 50, 150)
    return canny

cap = cv2.VideoCapture('test6.mp4')
forback = cv2.createBackgroundSubtractorKNN()


while(cap.isOpened()):
    _, frame = cap.read()
    fgmask = forback.apply(canny(frame))

    cv2.imshow('Frame', canny(frame))
    cv2.imshow('FG mask Frame', canny(fgmask))


    if cv2.waitKey(1) ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()