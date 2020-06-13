# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:22:27 2020

@author: nuwan abeynayake
"""

import cv2
import facerec as fr

cap = cv2.VideoCapture(0)

count =0

while True:
    ret,test_img = cap.read()
    faces_detected , gray_img = fr.faceDetection(test_img)
    
    if not ret:
        continue
    for face in faces_detected:
 
        
        cv2.imwrite("F:\Face Recognition Bot/forgithub/Training Images/6/frame%d.jpg"% count , test_img)
        count+=1
        
    resized_img = cv2.resize(test_img, (1000,700))
    cv2.imshow('face detection tutorial', resized_img)
    if cv2.waitKey(10) ==ord('q'):
        break
    
    
cap.release()
cv2.destroyAllWindows



