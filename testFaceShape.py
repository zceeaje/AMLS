from numpy import loadtxt,zeros,argmax,sum;from numpy.random import shuffle
from keras.utils import to_categorical; from keras.models import load_model
import imutils, numpy as np, time, dlib, cv2
from scipy.spatial import distance as dist
from imutils import face_utils;import glob
rootFolder = "c:\\Users\\daron\\Documents\\AMLS\\newData\\B1\\"
modelFile = "faceModel.h5";myModel = load_model(modelFile)
shape_predictor= "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

def testFiles(fileList,correctAnswer):
	corrects = 0; incorrects = 0
	for file in fileList:
		img = cv2.imread(file)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		rects = detector(gray, 0)
		if	len(rects) == 1:
			shape = predictor(gray, rects[0])
			shape = face_utils.shape_to_np(shape)
			jaw = shape[0:17]
			left = min(jaw[:,0]);right=max(jaw[:,0])
			top = min(jaw[:,1]);bottom=max(jaw[:,1])
			width = right-left;height = bottom-top
			biggest = max(width,height)		
			jaw[:,0] = jaw[:,0]-left
			jaw[:,1] = jaw[:,1]-top
			for point in range(jaw.shape[0]):
				for coord in range(jaw.shape[1]):
					jaw[point,coord] = int(jaw[point,coord] * 100/biggest)
			Xarray = jaw.reshape([1,34])
			if myModel.predict_classes(Xarray)[0] == correctAnswer:
				corrects +=1
			else:
				incorrects +=1
	return corrects, incorrects

for faceShape in range(5):
	fileList= glob.glob(rootFolder + "test\\" + str(faceShape)+"\\*.png")
	print("Testing files for faceShape %d (%d items)\n"%(faceShape, len(fileList)))
	corrects, incorrects = testFiles(fileList,faceShape)
	print("Correct: %d, Incorrect: %d"%(corrects,incorrects))