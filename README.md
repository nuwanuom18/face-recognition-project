# face-recognition-project
This project provides real time face recognition using OpenCV-Python. 

# How it works
## put images to recognize
YouHow to can put images to project manulally  or can take images using webcam (if you run videotoimg.py
file , your webcam is opened and it takes pictures continuously (you do not have to press buttons to capture images , it automatically captures)

if it detects a face and save them in a folder) then you have to provide a name for this person then to process the all images in seperate image folders under img folder run tester.py. Now it creates tranningData.yml file.

All set !

## recognize faces in webcam video
Now run videoTester.py file. Your webcam is opened and if it detects face/faces can be recognized by reading tranningData.yml, it shows that face/faces from a rectangle with name that you provided and confidence level for each recognition is shown in console.

## recognize faces in images
Also you can recognize face in the images as well. Add path of the image that you have recognize faces to recimages.py and run it. Now it reads the tranningData.yml file and mark faces from a rectangle with name that you provided and confidence level for each recognition is shown in console.
