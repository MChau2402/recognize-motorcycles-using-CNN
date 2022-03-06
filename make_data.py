import numpy as np
import cv2
import time
import os

label = "0_khongphaixe"

cap = cv2.VideoCapture(0)

i=0
while(True):
    i+=1
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.resize(frame, dsize=None, fx=1, fy=1)

    cv2.imshow('frame', frame)
    if i>=60:
        print("Số ảnh capture = ", i-60)
        if not os.path.exists('data/' + str(label)):
            os.mkdir('data/' + str(label))


        cv2.imwrite('data/' + str(label) + "/" + str(i-60) + ".png", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()