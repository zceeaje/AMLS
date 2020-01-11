# Import necessary libraries
from numpy import loadtxt,argmax,zeros,sum, set_printoptions; from numpy.random import shuffle
from keras.models import Sequential;from keras.layers import Dense, Dropout, LeakyReLU
from keras.utils import to_categorical; import argparse
from keras.layers import Dense, Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras import models
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import imutils, numpy as np, time, dlib, cv2
from scipy.spatial import distance as dist
from imutils import face_utils; import glob
from config import *

#epoch 50
def generateA2Data(use):
    shape_predictor= "A2"+dl+"shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    rootFolder = dataFolder + "A2" + dl
    codes = {"yes":0,"no":1}
    smilearrays = {}
    for smile in ["yes","no"]:
        files = glob.glob(rootFolder+use+dl+smile+dl+"*.jpg")
        smilearrays[smile] = np.zeros([len(files),41])
        for f in range(len(files)):
            img = cv2.imread(files[f])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 0)
            if	len(rects) == 1:
                shape = predictor(gray, rects[0])
                shape = face_utils.shape_to_np(shape)
                mouth = shape[mStart:mEnd]
                left = min(mouth[:,0]);right=max(mouth[:,0])
                top = min(mouth[:,1]);bottom=max(mouth[:,1])
                width = right-left
                height = bottom-top
                biggest = max(width,height)
                mouth[:,0] = mouth[:,0]-left
                mouth[:,1] = mouth[:,1]-top
                for point in range(mouth.shape[0]):
                    for coord in range(mouth.shape[1]):
                        mouth[point,coord] = int(mouth[point,coord] * 100/biggest)
                flatmouth = mouth.reshape(1,40)
                withlabel = np.insert(flatmouth,0,codes[smile])
                smilearrays[smile][f,:] = withlabel[:]
    return(np.vstack([smilearrays["yes"],smilearrays["no"]]))

def preProcessDataA2():
    myData = {}
    for use in ["train","validate","test"]:
        myData[use] = generateA2Data(use)
    return(myData["train"],myData["validate"],myData["test"])

def buildA2():
    sizeHL = [250, 200,150,100,50]
    numHL = len(sizeHL)
    actFunc = 'relu'
    dropout = 0.2
    myOptimizer = 'adam'
    myLoss= 'binary_crossentropy'
    model = Sequential()
    model.add(Dense(sizeHL[0], input_dim=40, activation=actFunc))
    for L in range(numHL-1):
        model.add(Dense(sizeHL[L+1], activation=actFunc))
        model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=myLoss, optimizer=myOptimizer, metrics=['accuracy'])
    return(model)

def trainA2(model_A2,A2trainData, A2valData):
    batchSize= 16
    numEpochs = 50
    Xtrain = A2trainData[:,1:]
    Ytrain = A2trainData[:,0]
    Xval = A2valData[:,1:]
    Yval = A2valData[:,0]
    history = model_A2.fit(Xtrain, Ytrain, validation_data = (Xval,Yval),
							epochs=numEpochs, batch_size=batchSize)
    return(history.history['accuracy'][49])

def testA2(model_A2, A2testData):
    print("Testing Neural Network A2")
    batchSize= 16
    Xtest = A2testData[:,1:]
    Ytest = A2testData[:,0]
    metrics = model_A2.evaluate(Xtest,Ytest,batch_size = batchSize)
    return(metrics[1])