# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:22:27 2020

@author: nuwan abeynayake
"""
import cv2
import facerec as fr

test_img = cv2.imread('F:\Face Recognition Bot/girl.jpg')
faces_detected,gray_img = fr.faceDetection(test_img)
print("faces_detected: ",faces_detected)

face_recognizer =cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('F:\Face Recognition Bot/trainingData.yml')

name = {0: 'sai pallavi',1:'Nuwan' ,5:'Sohan'}

for face in faces_detected:
    (x,y,w,h) =face
    roi_gray = gray_img[y:y+h,x:x+h]
    label, confidence = face_recognizer.predict(roi_gray)
    print("confidence: ",confidence)
    print("label: ",label)
    
    fr.draw_rect(test_img, face)
    predicted_name = name[label]
    if(confidence>75):
        continue
    fr.put_text(test_img, predicted_name, x, y)

resized_img = cv2.resize(test_img,(720,560))
cv2.imshow("face detection tutorial", test_img)
cv2.waitKey(0);
cv2.destroyAllWindows