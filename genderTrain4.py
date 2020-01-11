#========Program to fulfill task A1 - to predict gender from celebrity photographs==========
#IMPORT NECESSARY LIBRARIES
import os;import sys;import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Input, Flatten, Dense,  Dropout, Activation, MaxPooling2D, AveragePooling2D, Conv2D
from keras.models import Model,Sequential
import matplotlib.pyplot as plt

#SET GLOBAL VARIABLES
pictureDimensions = (218,178);picture3D = (218,178,3);batchSize = 32
rootFolder = "/Users/faymita29/Desktop/AMLS/AMLS_19-20_SN16015140/Datasets/A1/"
sets = ["train","test","validate"];folderNames = {}
for mySet in sets: folderNames[mySet]= rootFolder + mySet
    
#DEFINE NEURAL NETWORK HYPERPARAMETERS
epochs=5;dropout=0.0

def prepareData():
#	dataGenerator = ImageDataGenerator()
	dataGenerator = ImageDataGenerator(rotation_range=40,width_shift_range=0.2,
		height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')
	trainGen = dataGenerator.flow_from_directory(folderNames["train"],target_size=pictureDimensions,
        batch_size=batchSize, class_mode='binary',shuffle=True,color_mode="rgb")
	validateGen = dataGenerator.flow_from_directory(folderNames["validate"],target_size=pictureDimensions,
        batch_size=batchSize, class_mode='binary',shuffle=True,color_mode="rgb")
	testGen = dataGenerator.flow_from_directory(folderNames["test"],target_size=pictureDimensions,
        batch_size=batchSize, class_mode='binary',shuffle=True,color_mode="rgb")
	return(trainGen,validateGen,testGen)
	
def buildConvNetwork():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=picture3D));model.add(Activation('relu'));model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(32, (3, 3)));model.add(Activation('relu'));model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3)));model.add(Activation('relu'));model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten());model.add(Dense(64));model.add(Activation('relu'));model.add(Dropout(dropout))
	model.add(Dense(1));model.add(Activation('sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return(model)
		
trainGen, validateGen, testGen = prepareData()
model = buildConvNetwork()
model.summary()
history = model.fit_generator(generator=trainGen,steps_per_epoch=trainGen.n/batchSize,epochs=epochs,
					validation_data=validateGen,validation_freq = 1, validation_steps=validateGen.n/batchSize)
model.save("Gender.h5")

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'validate'], loc='upper left')
plt.show()
