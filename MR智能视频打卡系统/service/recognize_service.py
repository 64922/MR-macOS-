import cv2
import numpy as np
import os
RECOGNIZER=cv2.face.LBPHFaceRecognizer_create()
PASS_CONF=45
path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'cascades')
path_new=os.path.join(path,'haarcascade_frontalface_default.xml')
# print(path_new)
FACE_CASCADE=cv2.CascadeClassifier(path_new)
def train(photos,labels):
    RECOGNIZER.train(photos,np.array(labels))
def found_face(gray_img):
    faces=FACE_CASCADE.detectMultiScale(gray_img,1.15,4)
    return len(faces)>0
def recognize_face(photo):
    label,confidence=RECOGNIZER.predict(photo)
    if confidence>PASS_CONF:
        return -1
    return label
