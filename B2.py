# Import necessary libraries
from numpy import loadtxt,argmax,zeros,sum, set_printoptions; from numpy.random import shuffle
from keras.models import Sequential;from keras.layers import Dense, Dropout, LeakyReLU
from keras.utils import to_categorical; import argparse
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras import models
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import numpy as np, time, cv2, glob, math
from config import *

#epoch 35
leftEyeCentreX = 205; leftEyeCentreY = 260
leftEyeCentre = (leftEyeCentreX,leftEyeCentreY)
pupilRadius = 7; irisRadius = 15
sizeROI = 548 # Number of pixels in region of interest (iris)
labels = {"black":"0","brown":"1","blue":"2","green":"3","grey":"4"}
rootFolder = dataFolder + "B2" + dl

def generateB2Data(use):
    eyearrays = {}
    labels = {"black":0,"brown":1,"blue":2,"green":3,"grey":4}
    for myClass in ["black","brown","blue","green","grey"]:
        targetFolder = rootFolder+use+dl+myClass+dl+"*.png"
        cartoonFiles=(glob.glob(rootFolder+use+dl+myClass+dl+"*.png"))
        eyearrays[myClass] = np.zeros([len(cartoonFiles),3*sizeROI+1])
        for f in range(len(cartoonFiles)):
            COL = np.array([],dtype=np.uint8)
            img = cv2.imread(cartoonFiles[f])
            fileStr = labels[myClass]
            for x in range(-irisRadius,irisRadius):
                for y in range(-irisRadius,irisRadius):
                    distance = math.sqrt((x*x)+(y*y))
                    if (distance > pupilRadius) and (distance < irisRadius):
                        COL = np.append(COL,img[leftEyeCentreY+y][leftEyeCentreX+x])
            withlabel = np.insert(COL,0,labels[myClass])
            eyearrays[myClass][f,:] = withlabel[:]
    return(np.vstack([eyearrays["black"],eyearrays["brown"],eyearrays["blue"],eyearrays["green"],eyearrays["grey"]]))

def preProcessDataB2():
    myData = {}
    for use in ["train","validate","test"]:
        myData[use] = generateB2Data(use)
    return(myData["train"],myData["validate"],myData["test"])

def buildB2():
    sizeHL = [200,150,100,50]
    numHL = len(sizeHL)
    actFunc = 'relu'
    dropout = 0.0
    myOptimizer = 'adam'
    myLoss= 'categorical_crossentropy'
    model = Sequential()
    model.add(Dense(sizeHL[0], input_dim=3*sizeROI, activation=actFunc))
    for L in range(numHL-1):
        model.add(Dense(sizeHL[L+1], activation=actFunc))
        model.add(Dropout(dropout))
    model.add(Dense(5, activation='softmax'))
    model.compile(loss=myLoss, optimizer=myOptimizer, metrics=['accuracy'])
    return(model)

def trainB2(model_B2,B2trainData,B2valData):
    batchSize= 16
    numEpochs = 35
    Xtrain = B2trainData[:,1:]
    Ytrain = to_categorical(B2trainData[:,0])
    Xval = B2valData[:,1:]
    Yval = to_categorical(B2valData[:,0])
    history = model_B2.fit(Xtrain, Ytrain, validation_data = (Xval,Yval),
							epochs=numEpochs, batch_size=batchSize,shuffle=True)
    return(history.history['accuracy'][34])

def testB2(model_B2, B2testData):
    print("Testing Neural Network B2")
    batchSize= 16
    Xtest = B2testData[:,1:]
    Ytest = to_categorical(B2testData[:,0])
    metrics = model_B2.evaluate(Xtest,Ytest,batch_size = batchSize)
    return(metrics[1])