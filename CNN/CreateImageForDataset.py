# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 22:34:06 2023

@author: 62811
"""

import cv2
import numpy as np
from keras.models import load_model
import os

widthImg = 200
heightImg = 300
cap = cv2.VideoCapture(1)
cap.set(10, 130)

# output_folder = 'king of spades'
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
    
# record_duration = 20
# start_time = cv2.getTickCount()  
# frame_count = 0

fps = cap.get(cv2.CAP_PROP_FPS)

model = load_model('C:/Users/62811/PCV/Big Project/CNN/ALLINONE.h5')

folder = r"C:/Users/62811/PCV/Big Project/CNN/Dataset/Training/Images"
classes = [i for i in os.listdir(folder)]

def predictedImg(image):
    predictions = model.predict(np.array([image]))
    class_index = int(np.argmax(predictions))
    className = classes[class_index]
    confidence = predictions[0, class_index]
    return confidence, className
    
def preLiminaryProcess(image):
    imgGray =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
    imgCanny = cv2.Canny(imgBlur,200,200)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThres = cv2.erode(imgDial,kernel,iterations=1)
    return imgThres

def contourizationFrame(image):
    biggest = np.array([])
    maxArea = 0
    contours, heir = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for i in contours:
        area = cv2.contourArea(i)
        #print(area)
        if area>5000:
            #cv2.drawContours(imgContour, i, -1, (255,0,0), 3)
            mySharpEnd = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02*mySharpEnd, True)
            if area>maxArea and len(approx) == 4:
                biggest = approx
                maxArea = area
    cv2.drawContours(imgContour, biggest, -1, (255,0,0), 20)
    return biggest

def rearrangeMatrix(sharpEnd):
    sharpEnd = sharpEnd.reshape((4,2))
    sharpEndNew = np.zeros((4,1,2),np.int32)
    add = sharpEnd.sum(1)
    #print("add", add)
    
    sharpEndNew[0] = sharpEnd[np.argmin(add)]
    sharpEndNew[3] = sharpEnd[np.argmax(add)]
    
    diff = np.diff(sharpEnd,axis=1)
    sharpEndNew[1] = sharpEnd[np.argmin(diff)]
    sharpEndNew[2] = sharpEnd[np.argmax(diff)]
    #print("New Point", sharpEndNew)
    return sharpEndNew
    
def warppingFrame(image, biggestcontour):
    biggest = rearrangeMatrix(biggestcontour)
    #print(biggest)
    matrixWarp1 = np.float32([biggest[0],biggest[1],biggest[2],biggest[3]])
    matrixWarp2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(matrixWarp1, matrixWarp2)
    imgOutput = cv2.warpPerspective(image, matrix, (widthImg, heightImg))
    return imgOutput



while True:
    success,img = cap.read()
    if not success:
        break
    img = cv2.resize(img, (widthImg, heightImg))
    imgContour = img.copy()
    
    imgThres = preLiminaryProcess(img)
    biggest = contourizationFrame(imgThres)
    
    if biggest.size !=0:    
        #print(biggest)
        imgWarped = warppingFrame(img, biggest)
        imgWarpedResize = cv2.resize(imgWarped, (200, 300))
        imgWarpedGray =cv2.cvtColor(imgWarpedResize,cv2.COLOR_BGR2GRAY)
        imgThres1 = cv2.adaptiveThreshold(imgWarpedGray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,67,11)
        normalizedImage = imgThres1/255.0
        normalizedImage = cv2.resize(normalizedImage,(128,128))
        cv2.imshow("warped",imgThres1)
        confidence, className = predictedImg(normalizedImage)
        if confidence > 0.85:
            cv2.putText(img, className, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)


        # frame_count += 1
        # imgFilename = os.path.join(output_folder, f'{output_folder}_{frame_count}.png')
        # cv2.imwrite(imgFilename, imgThres1) 


        #Memeriksa apakah perekaman sudah selesai
        # current_time = cv2.getTickCount()
        # elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
        # if elapsed_time >= record_duration:
        #     break
      
    else:
        cv2.putText(img, 'No Card Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
    verticalAppendedImg = np.vstack((imgThres))
    cv2.imshow("Contour", verticalAppendedImg)                                   
    cv2.imshow("Result", img)

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()