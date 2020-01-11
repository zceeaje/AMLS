#========Program to fulfill task A1 - to predict gender from celebrity photographs==========
#IMPORT NECESSARY LIBRARIES
import os, sys, glob,cv2 ;import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Input, Flatten, Dense,  Dropout, Activation, MaxPooling2D, AveragePooling2D, Conv2D
from keras.models import Model,Sequential, load_model
from config import *

#epoch 10
batchSize = 32 #DEFINE A1 GLOBAL NEURAL NETWORK HYPERPARAMETERS

#build simple CNN network for binary data
def buildA1():
    picture3D = (218,178,3); dropout=0.0 #DEFINE BUILD RELATED NEURAL NETWORK HYPERPARAMETERS
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=picture3D))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return(model)
#epoch 15
def prepareDataA1():
    pictureDimensions = (218,178)
    rootFolder = dataFolder + "A1" + dl
    sets = ["train","test","validate"]
    folderNames = {}
    for mySet in sets:
        folderNames[mySet]= rootFolder + mySet
    dataGenerator = ImageDataGenerator()
    trainGen = dataGenerator.flow_from_directory(folderNames["train"],target_size=pictureDimensions,
        batch_size=batchSize, class_mode='binary',shuffle=True,color_mode="rgb")
    validateGen = dataGenerator.flow_from_directory(folderNames["validate"],target_size=pictureDimensions,
        batch_size=batchSize, class_mode='binary',shuffle=True,color_mode="rgb")
    return(trainGen,validateGen)

def trainA1(model_A1):
    #DEFINE BUILD RELATED NEURAL NETWORK HYPERPARAMETERS
    epochs = 10
    trainGen, validateGen = prepareDataA1()
    history = model_A1.fit_generator(generator=trainGen,steps_per_epoch=trainGen.n/batchSize,epochs=epochs,validation_data=validateGen,validation_freq = 1, validation_steps=validateGen.n/batchSize)
    return(history.history['accuracy'][9])

def testA1(model_A1):
    print("Testinging Neural Network A1")
    right, wrong = {},{}
    genderCodes= {"female":0,"male":1}
    for gender in ["female","male"]:
        files = glob.glob(dataFolder + "A1" + dl + "test" + dl + gender + dl + "*.jpg")
        right[gender] = 0
        wrong[gender] = 0
    for f in files:
        testArray = img_to_array(load_img(f))
        testArray = np.expand_dims(testArray,axis=0)
        prediction = model_A1.predict_classes(testArray)[0][0]
        if prediction == genderCodes[gender]: 
            right[gender] +=1
        else:
            wrong[gender] +=1
    totalRight = sum(right.values())
    totalWrong = sum(wrong.values())
    totalAccuracy = totalRight/(totalRight + totalWrong)
    return(totalAccuracy)