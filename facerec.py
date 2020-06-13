# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:22:27 2020

@author: nuwan abeynayake
"""

import cv2
import os
import numpy as np

def faceDetection(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = cv2.CascadeClassifier('F:\Face Recognition Bot/haarcascade_frontalface_default.xml')
    faces = face.detectMultiScale(gray_img, scaleFactor =1.3 , minNeighbors = 5)
    return faces,gray_img

def labels_for_training_data(directory):
    faces=[]
    facesID=[]
    
    for path,subdirnames,filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith("."):
                print("skipping system files")
                continue
            id = os.path.basename(path)
            img_path = os.path.join(path, filename)
            print("image_path: ",img_path)
            print("id: ",id)
            test_img = cv2.imread(img_path)
            if test_img is None:
                print("Image not loaded properly")
                continue
            faces_rect, gray_img= faceDetection(test_img)
            if len(faces_rect)!=1:
                continue
            (x,y,w,h) = faces_rect[0]
            roi_gray = gray_img[y:y+w,x:x+h]# taking only the face
            faces.append(roi_gray)
            facesID.append(int(id))
    return faces,facesID

def train_classifier(faces,facesID):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(facesID))
    return face_recognizer

def draw_rect(test_img,face):
    (x,y,w,h) =face
    cv2.rectangle(test_img, (x,y), (x+w,y+h),(255,0,0),thickness=5)
    
    
def put_text(test_img,text , x,y):
    cv2.putText(test_img,text, (x,y), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0),6)
    #cv2.FONT_HERSHEY_DUPLEX
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    