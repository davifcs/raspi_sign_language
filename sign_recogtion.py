# Import packages
import os
import cv2
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import argparse
import sys
from keras.models import load_model

# Set up camera constants
IM_WIDTH = 640
IM_HEIGHT = 480

# Initialize Picamera 
camera = PiCamera()
camera.resolution = (IM_WIDTH,IM_HEIGHT)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
rawCapture.truncate(0)

# Import trainned model
model=load_model('sign.h5')		


signs = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
		"N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", ""]

def processing(frame):
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame_resized = cv2.resize(frame_gray,(28,28))
	frame_final = frame_resized.reshape(1,28,28,1)
	
	return frame_final

def classify(frame_processed):
    ans = model.predict(frame_processed)[0]
     
    if (1 in ans) == True:
        result = signs[np.where(ans == 1)[0][0]]
        return result
    else:
        result = signs[26]
        return 
     

for frame in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
    
    frame = np.copy(frame.array)
    frame_processed = processing(frame)
    sign = classify(frame_processed)
    cv2.putText(frame, sign, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
    cv2.imshow('Object detector', frame)

    if cv2.waitKey(1) == ord('q'):
        break

    rawCapture.truncate(0)

camera.close()
cv2.destroyAllWindows()

