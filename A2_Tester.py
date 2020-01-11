from numpy import loadtxt,zeros,argmax,sum;from numpy.random import shuffle
from keras.utils import to_categorical; from keras.models import load_model
import imutils, numpy as np, time, dlib, cv2
from scipy.spatial import distance as dist
from imutils import face_utils;import glob

#load the model for testing 
modelFile = "Smile.h5";myModel = load_model(modelFile)
shape_predictor= "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
#cv2.namedWindow('img',cv2.WINDOW_AUTOSIZE)
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

#preprocess the data exactly the same as train and validate data
def testFiles(fileList,correctAnswer):
	corrects = 0; incorrects = 0
	for file in fileList:
		img = cv2.imread(file)
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
			#cv2.rectangle(img,(left,top),(right,bottom),(255,255,255),2)
			#for i in range(mouth.shape[0]-1):
				#cv2.line(img,(mouth[i][0],mouth[i][1]),(mouth[i+1][0],mouth[i+1][1]),(255,255,255),1)
			Xarray = mouth.reshape([40])
			Xarray = Xarray.reshape([1,40])
			if myModel.predict_classes(Xarray)[0] == correctAnswer:
				corrects +=1
			else:
				incorrects +=1
		#cv2.imshow('img',img)
		#cv2.waitKey(1)
	return corrects, incorrects

smileFiles=(glob.glob("/Users/faymita29/Downloads/Datasets/A2/test/yes/*.jpg"))
noSmileFiles=(glob.glob("/Users/faymita29/Downloads/Datasets/A2/test/no/*.jpg"))
truePositive,falseNegative = testFiles(smileFiles,0)
trueNegative, falsePositive = testFiles(noSmileFiles,1)
total = truePositive + trueNegative + falsePositive + falseNegative
print(truePositive,trueNegative,falsePositive, falseNegative)
print("Accuracy: %0.1f%%"%((truePositive+trueNegative)*100/total))

#cv2.destroyAllWindows()
