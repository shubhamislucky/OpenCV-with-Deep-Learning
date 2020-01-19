import cv2 as cv
import numpy as np

img = cv.imread('images/backpack.jpg')

all_rows = open('models/synset_words.txt').read().strip().split('\n')

classes = [r[r.find(' ') + 1:] for r in all_rows]

net = cv.dnn.readNetFromCaffe('models/bvlc_googlenet.prototext', 'models/bvlc_googlenet.caffemodel')
blob = cv.dnn.blobFromImage(img, 1, (224,224)) # enter 1 after img so that it doesn't resize it
net.setInput(blob)

outp = net.forward()
# print(outp)

idx = np.argsort(outp[0])[::-1][:5]

for (i, obj_id) in enumerate(idx):
    print('{}. {} ({}): Probability {:.3}%'.format(i+1, classes[obj_id], obj_id, outp[0][obj_id]*100 ))


# for (i, c) in enumerate(classes):
#     if i == 4:
#         break
#     print(i, c)

cv.imshow('backpack', img)
cv.waitKey(0)
cv.destroyAllWindows()