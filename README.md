# gender_detection

This is a simple machine learning application of gender detection on video or photo . The following repository contains 
the model and the weights trained on almost 25000 64x64x1 images which is provided by UTK faces. 
The model is quite simple and did well when the image to process was deducted to 64x64 from 250x250 which was 
previously planned. The repository contains notebook for the actual model building with a notebook demonstrating
the tuning process which has been performed by adequate use of the keras tuner.
The model has default input of a video call which can be found on youtube .
The cv_script is the main script which is run to demonstrate the application . 
This experiment is a good experience provider. I wasn't able to increase the value accuracy to more then 92%. Any suggestions are welcome. 
