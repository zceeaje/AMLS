#========Program to fulfill task AI - to predict gender from celebrity photographs==========
#IMPORT NECESSARY LIBRARIES
import os;import sys;import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Input, Flatten, Dense,  Dropout, Activation, MaxPooling2D, AveragePooling2D, Conv2D
from keras.models import Model,Sequential
#SET GLOBAL VARIABLES
pictureDimensions = (500,500);picture3D = (500,500,3);batchSize = 32
rootFolder = "newData\\B2\\";sets = ["train","test","validate"];folderNames = {}
for mySet in sets: folderNames[mySet]= rootFolder + mySet
#DEFINE NEURAL NETWORK HYPERPARAMETERS
epochs=10;dropout=0.1

def prepareData():
	dataGenerator = ImageDataGenerator()
#	dataGenerator = ImageDataGenerator(rotation_range=40,width_shift_range=0.2,
#		height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
	trainGen = dataGenerator.flow_from_directory(folderNames["train"],target_size=pictureDimensions,
        batch_size=batchSize, class_mode='categorical',shuffle=True,color_mode="rgb")
	validateGen = dataGenerator.flow_from_directory(folderNames["validate"],target_size=pictureDimensions,
        batch_size=batchSize, class_mode='categorical',shuffle=True,color_mode="rgb")
	testGen = dataGenerator.flow_from_directory(folderNames["test"],target_size=pictureDimensions,
        batch_size=batchSize, class_mode='categorical',shuffle=True,color_mode="rgb")
	return(trainGen,validateGen,testGen)
	
def buildConvNetwork():
	model = Sequential()
	model.add(Flatten(input_dim=picture3D));
	model.add(Dense(32));model.add(Activation('relu'));model.add(Dropout(dropout))
	model.add(Dense(5));model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return(model)
		
trainGen, validateGen, testGen = prepareData()
model = buildConvNetwork()
#model.summary()
model.fit_generator(generator=trainGen,steps_per_epoch=trainGen.n/batchSize,epochs=epochs,
					validation_data=validateGen,validation_freq = 1, validation_steps=validateGen.n/batchSize)
model.save("Faceshape.h5")
