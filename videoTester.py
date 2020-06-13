# -*- coding: utf-8 -*-


import cv2
import facerec as fr


face_recognizer =cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('F:\Face Recognition Bot/forgithub/trainingData.yml')

name = {0: 'sai',1:'Nuwan', 5: 'Sohan'}
cap = cv2.VideoCapture(0)
count=0
while True:
    ret,test_img = cap.read()
    faces_detected , gray_img = fr.faceDetection(test_img)
    
    # for (x,y,w,h) in faces_detected:
    #     cv2.rectangle(test_img, (x,y), (x+w,y+h), (255,0,0),thickness = 7)
    resized_img = cv2.resize(test_img,(1000,700))
    cv2.imshow('face detection tutorial', resized_img)
    cv2.waitKey(10)         
    
    for face in faces_detected:
        (x,y,w,h) =face
        roi_gray = gray_img[y:y+h,x:x+h]
        label, confidence = face_recognizer.predict(roi_gray)
        print("confidence: ",confidence)
        print("label: ",label)
        
        
        predicted_name = name[label]
        if(confidence>81):
            continue
        
        
        # if predicted_name=='Nuwan':
        #     cv2.imwrite("F:\Face Recognition Bot\Training Images/1/frame%d.jpg"% count , test_img)
        #     count+=1
        fr.draw_rect(test_img, face)
        cv2.rectangle(test_img, (x,y), (x+w,y+h), (255,0,0),thickness = 1)
        fr.put_text(test_img, predicted_name, x, y)
        
        
    resized_img = cv2.resize(test_img,(1000,700))
    cv2.imshow('face detection tutorial', resized_img)
    if cv2.waitKey(10) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows
