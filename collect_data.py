import cv2 
import os 
video = cv2.VideoCapture(0)
DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

classes = ['1', '2', '3', '4', '5']
num_images = 200
i = 0
dir = 0
collecting = False
while True:
    image = classes[dir]
    ret, frame = video.read()
    if not os.path.exists('data/{}'.format(image)):
        os.mkdir('data/{}'.format(image))
    if collecting:
        cv2.imwrite('data/{}/{}.jpg'.format(image, i), frame)
        i += 1 
    else:
        cv2.putText(frame, "Press Q to Collect Number {}".format(image), (20, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            collecting = True
    if i == num_images:
        collecting = False
        i = 0
        dir = dir + 1 
    if dir == len(classes):
        break
    cv2.imshow('Image', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break