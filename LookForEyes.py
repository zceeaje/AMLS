import imutils, numpy as np, time, dlib, cv2
from scipy.spatial import distance as dist
from imutils import face_utils;import glob

shape_predictor= "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

cv2.namedWindow('img',cv2.WINDOW_AUTOSIZE)

celebFiles=(glob.glob("/Users/faymita29/Desktop/AMLS/AMLS_19-20_SN16015140/Datasets/B2/*/*/*.png"))
for file in celebFiles:
	img = cv2.imread(file)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		cv2.fillPoly(img, [shape[36:41]],(255,0,255))
		cv2.polylines(img,[shape[36:41]],True,(255,255,255),2)
		cv2.fillPoly(img, [shape[42:47]],(255,0,255))
		cv2.polylines(img,[shape[42:47]],True,(255,255,255),2)
		cv2.polylines(img,[shape[0:16]],False,(255,0,255),5)

		cv2.imshow('img',img)
	cv2.waitKey(0)
cv2.destroyAllWindows()

