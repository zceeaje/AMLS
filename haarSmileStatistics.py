import numpy as np; import cv2; import glob
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile1.xml')

smileFiles=(glob.glob("/Users/faymita29/Desktop/AMLS/AMLS_19-20_SN16015140/Datasets/A2/train/yes/*.jpg"))
noSmileFiles=(glob.glob("/Users/faymita29/Desktop/AMLS/AMLS_19-20_SN16015140/Datasets/A2/train/no/*.jpg"))

ActualYesPredictYes = 0
ActualYesPredictNo = 0
ActualNoPredictYes = 0
ActualNoPredictNo = 0
numPhotos = len(smileFiles) + len(noSmileFiles)

for file in smileFiles:
	img = cv2.imread(file)
	print(".",end='',flush=True)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.1, 5)	
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		smiles = smile_cascade.detectMultiScale(roi_gray, 1.1, 35)
		if len(smiles)==0:
			ActualYesPredictNo +=1
		else:
			ActualYesPredictYes +=1

for file in noSmileFiles:
	img = cv2.imread(file)
	print(".",end='',flush=True)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.1, 5)	
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		smiles = smile_cascade.detectMultiScale(roi_gray, 1.1, 35)
		if len(smiles)==0:
			ActualNoPredictNo +=1
		else:
			ActualNoPredictYes +=1
print("%d photos checked"%(numPhotos))
print(ActualYesPredictYes,ActualYesPredictNo,ActualNoPredictYes,ActualNoPredictNo)