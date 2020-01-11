import math, cv2, glob,sys; import numpy as np
from keras.models import load_model

modelFile = "EyeColour.h5"
myModel = load_model(modelFile)
leftEyeCentreX = 205; leftEyeCentreY = 260
leftEyeCentre = (leftEyeCentreX,leftEyeCentreY)
pupilRadius = 7; irisRadius = 15
labels = {"black":0,"brown":1,"blue":2,"green":3,"grey":4}
def testFiles(fileList,correctAnswer):
	corrects = 0; incorrects = 0
	for file in fileList:
		img = cv2.imread(file)
		Xarray = []
		for x in range(-irisRadius,irisRadius):
			for y in range(-irisRadius,irisRadius):
				distance = math.sqrt((x*x)+(y*y))
				if (distance > pupilRadius) and (distance < irisRadius):
					COL = img[leftEyeCentreY+y][leftEyeCentreX+x]
					#print(int(COL[0]))
					Xarray.append(int(COL[0]));Xarray.append(int(COL[1]));Xarray.append(int(COL[2]))
		Xarray = np.array(Xarray)
		Xarray = np.expand_dims(Xarray, axis=0)
		prediction = myModel.predict_classes(Xarray)[0]
		if	prediction == correctAnswer:
			corrects +=1
		else:
			incorrects +=1
	return corrects, incorrects
totalCorrects = 0;totalIncorrects = 0

for myClass in ["black","brown","blue","green","grey"]:
	eyeFiles=(glob.glob("/Users/faymita29/Desktop/AMLS/AMLS_19-20_SN16015140/Datasets/B2/test/"+myClass+"/*.png"))
	#print(eyeFiles)
	corrects, incorrects = testFiles(eyeFiles,labels[myClass])
	totalCorrects += corrects; totalIncorrects +=incorrects
	pcCorrect = corrects/(corrects+incorrects); pcIncorrect = 1 - pcCorrect
    
	print("class: %s, correct: %d (%0.1f%%), incorrect: %d (%0.1f%%)"%
		(myClass,corrects,100*pcCorrect,incorrects,100*pcIncorrect))
    
print("Overall: %d correct, %d incorrect, accuracy %0.1f%%"%
		(totalCorrects, totalIncorrects, 100*totalCorrects/(totalCorrects+totalIncorrects)))
