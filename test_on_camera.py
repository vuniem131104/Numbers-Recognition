import cv2 
import pickle
from cvzone import HandTrackingModule
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
detector = HandTrackingModule.HandDetector()
video = cv2.VideoCapture(0)
model_pickle = pickle.load(open('model.p', 'rb'))
model = model_pickle['model']
while True:
    ret, frame = video.read()
    if ret:
        hands, frame = detector.findHands(frame)
        if hands:
            for hand in hands:
                data = [[]]
                lmList = hand['lmList']
                bbox = hand['bbox']
                x, y, w, h = [int(v) for v in bbox]
                for lm in lmList:
                    data[0].append(lm[0])
                    data[0].append(lm[1])
                data = np.asarray(data)
                y_pred = model.predict(data)[0]
                cv2.putText(frame, str(y_pred), (x + w - 20, y - 25), cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0), 2)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
video.release()
cv2.destroyAllWindows()