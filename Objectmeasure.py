#https://www.youtube.com/watch?v=tk9war7_y0Q
import cv2
import numpy as np
import utlis

webcam = True
path = '1.jpg'
cap = cv2.VideoCapture(1)
cap.set(10,160)
cap.set(3,640)
cap.set(4,480)
scale = 3
wP = 210*scale #width of a4 paper
hP = 297*scale #Height of a4 paper


while True:
    if webcam: success, img = cap.read()
    else: img = cv2.imread(path)

    imgCont, conts = utlis.getContours(img,minArea=5000, filter=4)

    if len(conts) !=0:
        biggest = conts[0][2]
        imgWarp = utlis.warpImg(img,biggest, wP,hP)

        imgCont2, conts2 = utlis.getContours(imgWarp, minArea=2000, filter=4,cThr = [50,50],draw=False)
        if len(conts) != 0:
            for obj in conts2:
                cv2.polylines(imgCont2, [obj[2]], True, (0, 255, 0), 2)
                nPoints = utlis.reorder(obj[2])
                nW = round(utlis.findDis(nPoints[0][0] // scale, nPoints[1][0] // scale), 1)
                nH = round(utlis.findDis(nPoints[0][0] // scale, nPoints[2][0] // scale), 1)
                cv2.arrowedLine(imgCont2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[1][0][0], nPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(imgCont2, (nPoints[0][0][0], nPoints[0][0][1]),
                                (nPoints[2][0][0], nPoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(imgCont2, '{}mm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv2.putText(imgCont2, '{}mm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
        cv2.imshow('A4', imgCont2)

    img = cv2.resize(img, (0,0), None,0.5,0.5)
    cv2.imshow('Result', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break