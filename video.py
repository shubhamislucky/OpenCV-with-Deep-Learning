import cv2 as cv
import numpy as np

# when we have a video file we use:
cap = cv.VideoCapture('images/witcher.mkv') # location of video file

# using webcam as input
# cap = cv.VideoCapture(0)

if cap.isOpened() == False:
    print('Can not open file or video stream')

while True:
    ret, frame = cap.read() # here ret is for return

    if ret == True:
        cv.imshow('movie', frame)

        if cv.waitKey(25) and 0xFF == ord('q'):
            break
    else:
        break


cap.release()
cv.destroyAllWindows()

