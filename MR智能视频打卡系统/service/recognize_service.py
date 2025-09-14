import cv2
import numpy as np
import os
RECOGNIZER=cv2.face.LBPHFaceRecognizer_create()
PASS_CONF=45
FACE_CASCADE=cv2.CascadeClassifier('/Users/chiral/PycharmProjects/PythonProject/MR智能视频打卡系统/cascades/haarcascade_frontalface_default.xml')
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
