#========Program to fulfill task A1 - to predict gender from celebrity photographs==========
#IMPORT NECESSARY LIBRARIES
import os;import sys;import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Input, Flatten, Dense,  Dropout, Activation, MaxPooling2D, AveragePooling2D, Conv2D
from keras.models import Model,Sequential
import matplotlib.pyplot as plt

#SET GLOBAL VARIABLES
pictureDimensions = (500,500);picture3D = (500,500,3)
rootFolder = "/Users/faymita29/Desktop/AMLS/AMLS_19-20_SN16015140/Datasets/B1/"
sets = ["train","test","validate"];folderNames = {}
for mySet in sets: folderNames[mySet]= rootFolder + mySet
    
#DEFINE NEURAL NETWORK HYPERPARAMETERS
epochs=10;dropout=0.0;batchSize = 10

def prepareData():
	dataGenerator = ImageDataGenerator()
	trainGen = dataGenerator.flow_from_directory(folderNames["train"],target_size=pictureDimensions,
        batch_size=batchSize, class_mode='categorical',shuffle=True,color_mode="rgb")
	validateGen = dataGenerator.flow_from_directory(folderNames["validate"],target_size=pictureDimensions,
        batch_size=batchSize, class_mode='categorical',shuffle=True,color_mode="rgb")
	return(trainGen,validateGen)

#build simple CNN network for categorical data
def buildConvNetwork():
	model = Sequential()
	model.add(Conv2D(24, (3, 3), activation='relu', padding='same', input_shape=picture3D));model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Conv2D(24, (3, 3), activation='relu', padding='same'));model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
	model.add(Conv2D(48, (3, 3), activation='relu', padding='same'));model.add(MaxPooling2D(pool_size=(2, 2), padding='same'));model.add(Conv2D(96, (3, 3), activation='relu', padding='same'));model.add(MaxPooling2D((2, 2), padding='same'))
	model.add(Flatten());model.add(Dense(64))
	model.add(Dense(5));model.add(Activation('softmax'))
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return(model)

#generate data from training and validating
trainGen, validateGen = prepareData()
model = buildConvNetwork()
model.summary()
history = model.fit_generator(generator=trainGen,steps_per_epoch=trainGen.n/batchSize,epochs=epochs,
					validation_data=validateGen,validation_freq = 1, validation_steps=validateGen.n/batchSize)

model.save("FaceShape.h5")

#plot graphs for all train and validate
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.ylim(0,1.2)
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

    



