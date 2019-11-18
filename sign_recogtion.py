# Import packages
import os
import cv2
import numpy as np
import pandas as pd
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import sys
from keras.models import load_model

# Set up camera constants
IM_WIDTH = 80
IM_HEIGHT = 64

storing = False

# Initialize Picamera 
camera = PiCamera()
camera.resolution = (IM_WIDTH,IM_HEIGHT)
camera.framerate = 5
rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
rawCapture.truncate(0)

# Import trainned model
model=load_model('signs-lang-recogtion.h5')		

# List for storing results
sign_result = []

signs = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M",
		"N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", ""]

def process(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = cv2.resize(frame_gray,(28,28))
    frame_final = frame_resized.reshape(1,28,28,1)
	
    return frame_final, frame_gray

def classify(frame_processed):
    ans = model.predict(frame_processed)[0]
     
    if (1 in ans) == True:
        label = np.where(ans == 1)
        result = signs[label[0][0]]
        return result, label
    else:
        label = 24
        result = signs[24]
        return result, label
     

for frame in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    frame = np.copy(frame.array)
    frame_processed, frame_gray = process(frame)
    sign, label = classify(frame_processed)
    cv2.putText(frame, sign, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.imshow('Object detector', frame)

    key = cv2.waitKey(1) 

    if key == ord('q'):
        break
    elif (key == ord('s') and storing == False):
        storing = True
    elif (key == ord('s') and storing == True):
        df_sign_result = pd.DataFrame(sign_result)
        df_sign_result.to_csv('sign_results.csv', index = False)
        storing = False

    if storing:
        ans = np.append(label, frame_gray)
        sign_result.append(ans)

    
    rawCapture.truncate(0)


camera.close()
cv2.destroyAllWindows()

