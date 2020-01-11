from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from keras.models import load_model
import glob, sys; import numpy as np; import cv2
picSize = (218,178);pic3D = (218,178,3)

#load the latest model saved from the trainging
myModel = load_model("Gender.h5")

womenfiles=(glob.glob("/Users/faymita29/Downloads/Datasets/A1/test/female/*.jpg"))
womenright = 0; womenwrong = 0; womenpredictions = []

#going through all the female files to test the model
for f in womenfiles:
	testImage = load_img(f)
	testArray = img_to_array(testImage)
	testArray = testArray.reshape((1,)+testArray.shape)
	prediction = myModel.predict_classes(testArray)[0][0]
	womenpredictions.append(prediction)
	if prediction == 0: 
		womenright +=1
	else:
		womenwrong +=1

menfiles=(glob.glob("/Users/faymita29/Downloads/Datasets/A1/test/male/*.jpg"))
menright = 0; menwrong = 0; menpredictions = []

#going through all the female files to test the model
for f in menfiles:
	testImage = load_img(f)
	testArray = img_to_array(testImage)
	testArray = testArray.reshape((1,)+testArray.shape)
	prediction = myModel.predict_classes(testArray)[0][0]
	menpredictions.append(prediction)
	if prediction == 1: 
		menright +=1
	else:
		menwrong +=1

#print all the results including for confusion matrix
print(womenright, womenwrong,menright,menwrong)
print("Overall accuracy: %0.2f%%"%(100*(menright+womenright)/(menright+menwrong+womenright+womenwrong)))
print("Women accuracy: %0.2f%%"%(100*womenright/(womenright+womenwrong)))
print("Men accuracy: %0.2f%%"%(100*menright/(menright+menwrong)))

#cv2.namedWindow('Test Pictures',cv2.WINDOW_AUTOSIZE)
#for f in range(len(menfiles)):
#	manImage = cv2.imread(menfiles[f]);womanImage = cv2.imread(womenfiles[f])
#	manText = ("woman","man")[menpredictions[f]];womanText = ("woman","man")[womenpredictions[f]]
#	manColor = ((0,0,255),(255,255,255))[menpredictions[f]]
#	womanColor = ((255,255,255),(0,0,255))[womenpredictions[f]]
#	cv2.putText(manImage,manText, (10,80),cv2.FONT_HERSHEY_SIMPLEX, 1.5, manColor, 2)
#	cv2.putText(womanImage,womanText, (10,80),cv2.FONT_HERSHEY_SIMPLEX, 1.5, womanColor, 2)
#	cv2.imshow('Test Pictures',np.hstack([womanImage,manImage]))
#	cv2.waitKey(200)
#cv2.destroyWindow('Test Pictures')