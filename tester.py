# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 11:22:27 2020

@author: nuwan abeynayake
"""

import facerec as fr



face,facesID = fr.labels_for_training_data('F:\Face Recognition Bot/forgithub/Training Images')
face_recognizer = fr.train_classifier(face, facesID)
face_recognizer.save('trainingData.yml')


    















            