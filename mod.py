import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np 



class prediction():
	def __init__(self,model,frame):
		self.model=model
		self.frame=frame
	def pred(self):
		return np.concatenate(self.model.predict(self.frame))[0]





