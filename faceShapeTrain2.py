# Import necessary libraries
from numpy import loadtxt,argmax,zeros,sum, set_printoptions; from numpy.random import shuffle
from keras.models import Sequential;from keras.layers import Dense, Dropout, LeakyReLU
from keras.utils import to_categorical; import argparse; import sys
# Set hyperparameters for Neural Network
verbose = True;sourceName = "faceData.csv";modelFile = "faceModel.h5"
sizeHL = [250, 200,150,100,50];numHL = len(sizeHL); actFunc = 'relu';numEpochs = 100;batchSize= 16; dropout = 0.2
myOptimizer = 'adam'; myLoss= 'binary_crossentropy'; validationProportion = 0.2

def loadSourceData():
	dataset = loadtxt(sourceName, delimiter=',');shuffle(dataset)
	X = dataset[:,1:];rawY = dataset[:,0];Y=to_categorical(rawY)
	numRows = X.shape[0];numInputs = X.shape[1];numClasses=Y.shape[1]
	if verbose: print("This data has",numRows,"rows,",numInputs,"inputs,",numClasses,"classes.")
	return X,Y,numRows,numInputs,numClasses

def buildNetwork():
	if verbose: print("Building Network with %d hidden layers..."%(numHL))
	model = Sequential()
	model.add(Dense(sizeHL[0], input_dim=numInputs, activation=actFunc))
	for L in range(numHL-1):
		model.add(Dense(sizeHL[L+1], activation=actFunc)); model.add(Dropout(dropout))
	model.add(Dense(numClasses, activation='sigmoid'))
	model.compile(loss=myLoss, optimizer=myOptimizer, metrics=['accuracy'])
	return(model)
	
def trainNetwork():
	if verbose: print("Training Network for %d epochs..."%(numEpochs))
	model.fit(X, Y, epochs=numEpochs, batch_size=batchSize,validation_split=validationProportion, verbose=verbose)
	loss, accuracy = model.evaluate(X, Y, verbose=verbose)
	return(model,accuracy)
	
def saveModel(model):
	model.save(modelFile); print("Saved model to disk.")
#=======Main code=====================================================================================================
X,Y,numRows,numInputs,numClasses = loadSourceData()
model = buildNetwork(); model,trainAccuracy = trainNetwork()
print("Training complete. Accuracy: %2.1f%%"%(trainAccuracy*100))
saveModel(model); print("Model saved as "+ modelFile)
