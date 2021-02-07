import cv2
import matplotlib.pyplot as plt

cap = cv2.VideoCapture('test4.mp4')

while(cap.isOpened()):
    _, image = cap.read()
    print(image.shape)
    plt.imshow(image)
    plt.show()
    cv2.imshow('Window Box Name', image)
    if cv2.waitKey(1) ==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()