import os
import numpy as np
import cv2
import matplotlib.pyplot as plt 
import mod
from mod import prediction
import tensorflow as tf 
from tensorflow import keras
model_path='gender_model.h5'
weights_path='gender_weights.h5'
final=keras.models.load_model(model_path)
final.load_weights(weights_path)

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface.xml')
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
gender_dictionary={0:'male',1:'female',100:'deciding'}
cap=cv2.VideoCapture('zoom.mp4')
val=100
v=0
eps=50
while True:
	ret,frame=cap.read()
	image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces=face_cascade.detectMultiScale(image,1.3,5)
	for (x,y,w,h) in faces:

		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
		roi_frame=image[y:y+h,x:x+w]
		resized_frame=cv2.resize(roi_frame,(64,64)).reshape(1,64,64,1)
		val=mod.prediction(final,resized_frame).pred()
		print(val)
		if val<=0.5:
			v=0
		else:
			v=1
		font=cv2.FONT_HERSHEY_SIMPLEX
		cv2.putText(frame,gender_dictionary[v],(x,y),font,1,(255,0,0),3,cv2.LINE_AA)
		##predict gender from model observing the resized_frame
	#font=cv2.FONT_HERSHEY_SIMPLEX
	#cv2.putText(frame,gender_dictionary[v],(200,200),font,1,(255,0,0),3,cv2.LINE_AA)
	cv2.imshow('frame',frame)
	
	if cv2.waitKey(1) & 0xFF==ord('q'):
		break
plt.imshow(frame)
cap.release()
cv2.destroyAllWindows()
print(type(gender_dictionary[0]))