#========Program to fulfill task A1 - to predict gender from celebrity photographs==========
#IMPORT NECESSARY LIBRARIES
import os, sys, glob, cv2 ;import numpy as np
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Input, Flatten, Dense,  Dropout, Activation, MaxPooling2D, AveragePooling2D, Conv2D
from keras.models import Model, Sequential, load_model
from config import *

#epoch 10
#DEFINE NEURAL NETWORK HYPERPARAMETERS
batchSize = 10

#build simple CNN network for categorical data
def buildB1():
    picture3D = (500,500,3)
    dropout=0.0
    model = Sequential()
    model.add(Conv2D(24, (3, 3), activation='relu', padding='same', input_shape=picture3D))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(24, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(48, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Conv2D(96, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Flatten());model.add(Dense(64))
    model.add(Dense(5));model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return(model)

def prepareDataB1():
    pictureDimensions = (500,500)
    rootFolder = dataFolder + "B1" + dl
    sets = ["train","test","validate"]
    folderNames = {}
    for mySet in sets:
        folderNames[mySet]= rootFolder + mySet
    dataGenerator = ImageDataGenerator()
    trainGen = dataGenerator.flow_from_directory(folderNames["train"],target_size=pictureDimensions,
        batch_size=batchSize, class_mode='categorical',shuffle=True,color_mode="rgb")
    validateGen = dataGenerator.flow_from_directory(folderNames["validate"],target_size=pictureDimensions,
        batch_size=batchSize, class_mode='categorical',shuffle=True,color_mode="rgb")
    return(trainGen,validateGen)

def trainB1(model_B1):
    epochs=10
    trainGen, validateGen = prepareDataB1()
    history = model_B1.fit_generator(generator=trainGen,steps_per_epoch=trainGen.n/batchSize,epochs=epochs,validation_data=validateGen,validation_freq = 1, validation_steps=validateGen.n/batchSize)
    return(history.history['accuracy'][9])

def testB1(model_B1):
    print("Testing Neural Network B1")
    labels = {"0":0,"1":1,"2":2,"3":3,"4":4}
    for myClass in ["0","1","2","3","4"]:
        faceShapeFiles=(glob.glob(dataFolder + "B1" + dl + "test" + dl + myClass + dl + "*.png"))
    corrects = 0
    incorrects = 0
    for file in faceShapeFiles:
        testImage = load_img(file)
        testArray = img_to_array(testImage)
        testArray = np.expand_dims(testArray, axis = 0)
        if model_B1.predict_classes(testArray) == labels[myClass]:
            corrects += 1
        else:
            incorrects +=1
    totalCorrects = 0
    totalIncorrects = 0
    totalCorrects += corrects
    totalIncorrects +=incorrects
    totalAccuracy = totalCorrects/(totalCorrects + totalIncorrects)
    return totalAccuracy