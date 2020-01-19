import cv2 as cv
import numpy as np

# in this program we will make a video classifier
# when we have a video file we use:
# cap = cv.VideoCapture('images/witcher.mkv') # location of video file

# using webcam as input
cap = cv.VideoCapture(0)

all_rows = open('models/synset_words.txt').read().strip().split('\n')

classes = [r[r.find(' ') + 1:] for r in all_rows]

net = cv.dnn.readNetFromCaffe('models/bvlc_googlenet.prototext', 'models/bvlc_googlenet.caffemodel')

if cap.isOpened() == False:
    print('Can not open file or video stream')

while True:
    ret, frame = cap.read() # here ret is for return

    blob = cv.dnn.blobFromImage(frame, 1, (224,224)) # enter 1 after img so that it doesn't resize it
    net.setInput(blob)
    outp = net.forward()

    r = 1
    for i in np.argsort(outp[0])[::-1][:5]:
        txt = ' "%s" probability "%.3f" ' % (classes[i], outp[0][i] * 100)
        
        # we use (0, 25 +40*r) to set the bottom left of text
        # we multipy it by r so that there is gap between all predictions of 40 spaces
        # 1 in putText signifies the scale of the text
        # then we set the colour of text
        # 2 is the thickness of lines to draw the text
        cv.putText(frame, txt, (0, 25 + 40*r), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        r += 1

    if ret == True:
        cv.imshow('movie', frame)

        if cv.waitKey(25) and 0xFF == ord('q'):
            break
    else:
        break


cap.release()
cv.destroyAllWindows()

