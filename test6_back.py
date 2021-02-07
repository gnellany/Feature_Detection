import cv2
import numpy as np

def canny(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255,0,0),10)
    return line_image

def region_of_intrest(image):
    height = image.shape[0]
    polygon = np.array([[(200,height),(1100, height),(550,250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygon,255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def lines(image):
    lines = cv2.HoughLinesP(image, 2, np.pi / 180, 100, np.array([]), minLineLength=20, maxLineGap=5)
    return lines

def image_copy(image):
    lane_image = np.copy(image)
    cropped_image = region_of_intrest(lane_image)
    line_image = display_lines(lane_image, lines(cropped_image))
    combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
    return combo_image

cap = cv2.VideoCapture('test6.mp4')

forback = cv2.createBackgroundSubtractorKNN()


while(cap.isOpened()):
    _, frame = cap.read()
    fgmask = forback.apply(canny(frame))

    cv2.imshow('FG mask Frame', image_copy(fgmask))

    if cv2.waitKey(1) ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()