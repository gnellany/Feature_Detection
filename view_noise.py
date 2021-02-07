import cv2

def canny(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

cap = cv2.VideoCapture(1)
kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
forback = cv2.bgsegm.createBackgroundSubtractorGMG()


while(cap.isOpened()):
    _, frame = cap.read()
    fgmask = forback.apply(frame)   
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernal)

    cv2.imshow('Frame', frame)
    cv2.imshow('FG mask Frame', canny(fgmask))


    if cv2.waitKey(1) ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()