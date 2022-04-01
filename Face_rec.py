import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('D:\A.I\Python_files\live_face_detection\Trainer\Trainer.yml')
cascadePath = 'D:\A.I\Python_files\live_face_detection\haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath)

font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

names = []

img = cv2.VideoCapture(0)
img.set(3, 640)
img.set(4, 480)

minW = 0.1 * img.get(3)
minH = 0.1 * img.get(4)

while True:
    ret, frame = img.read()
    frame = cv2.flip(frame, -1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize =(int(minW), int(minH)))
    
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0, 255), 2)

        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if (confidence < 100):
            id = names[id]
            confidence = ' {0}%'.format(round(100 - confidence))
        else: 
            id = 'unknown'
            confidence = ' {0}%'.format(round(100 - confidence))

        cv2.putText(frame, str(id), (x+5, y-5), font, 1, (255,255,255), 2)
        cv2.putText(frame, str(confidence), (x+5, y+h-5), font, 1, (255,255,0), 1)
    cv2.imshow('Camera', frame)

    k = cv2.waitKey(10) & 0xff 
    if k == 27:
        break

print('\n [INFO] Exiting Program and Cleanup stuff')
img.release()
cv2.destroyAllWindows()