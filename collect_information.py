import cv2 
from cvzone import HandTrackingModule
import os
import pickle
detector = HandTrackingModule.HandDetector()
DATA_DIR = './data'
data = []
labels = []

for dir in os.listdir(DATA_DIR):
    for image_path in os.listdir(os.path.join(DATA_DIR, dir)):
        data_aux = []
        image = cv2.imread(os.path.join(DATA_DIR, dir, image_path))
        hands, image = detector.findHands(image)
        if hands:
            hand = hands[0]
            lmList = hand['lmList']
            for lm in lmList:
                data_aux.append(lm[0])
                data_aux.append(lm[1])
            data.append(data_aux)
            labels.append(int(dir))

f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
                
