import numpy as np; import cv2; import glob
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from sklearn.metrics import confusion_matrix

modelFile = "FaceShape1.h5"
myModel = load_model(modelFile)
labels = {"0":0,"1":1,"2":2,"3":3,"4":4}
trueYs = []
predYs = []

def testFiles(fileList,correctAnswer):
    global trueYs, predYs
    corrects = 0
    incorrects = 0
    for file in fileList:
        testImage = load_img(file)
        testArray = img_to_array(testImage)
        testArray = np.expand_dims(testArray, axis = 0)
        prediction = myModel.predict_classes(testArray)
        trueYs.append(correctAnswer)
        predYs.append(prediction)
        if prediction == correctAnswer:
            corrects += 1
        else:
            incorrects +=1
    return corrects, incorrects

totalCorrects = 0
totalIncorrects = 0


for myClass in ["0","1","2","3","4"]:
	faceShapeFiles=(glob.glob("/Users/faymita29/Downloads/Datasets/B1/test/"+myClass+"/*.png"))
	#print(eyeFiles)
	corrects, incorrects = testFiles(faceShapeFiles,labels[myClass])
	totalCorrects += corrects; totalIncorrects +=incorrects
	pcCorrect = corrects/(corrects+incorrects); pcIncorrect = 1 - pcCorrect
    
	print("class: %s, correct: %d (%0.1f%%), incorrect: %d (%0.1f%%)"%
		(myClass,corrects,100*pcCorrect,incorrects,100*pcIncorrect))
    
print("Overall: %d correct, %d incorrect, accuracy %0.1f%%"%
		(totalCorrects, totalIncorrects, 100*totalCorrects/(totalCorrects+totalIncorrects)))

cm = confusion_matrix(trueYs,predYs)
print(cm)

