import imutils, numpy as np, time, dlib, cv2
from scipy.spatial import distance as dist
from imutils import face_utils;import glob

def smile(mouth):
    A = dist.euclidean(mouth[3], mouth[9])
    B = dist.euclidean(mouth[2], mouth[10])
    C = dist.euclidean(mouth[4], mouth[8])
    avg = (A+B+C)/3
    D = dist.euclidean(mouth[0], mouth[6])
    mar=avg/D
    return mar

shape_predictor= "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
cv2.namedWindow('img',cv2.WINDOW_AUTOSIZE)

celebFiles=(glob.glob("/Users/faymita29/Desktop/AMLS/AMLS_19-20_SN16015140/Datasets/A2/train/yes/*.jpg"))
for file in celebFiles:
	img = cv2.imread(file)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	for rect in rects:
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		mouth = shape[mStart:mEnd]
		cv2.fillPoly(img, pts =[shape[48:59]], color=(0,0,255))
		cv2.fillPoly(img, pts =[shape[60:67]], color=(255,255,255))
		#for i in range(mouth.shape[0]-1):
		#	cv2.line(img,(mouth[i][0],mouth[i][1]),(mouth[i+1][0],mouth[i+1][1]),(0,0,0),1)
		cv2.imshow('img',img)
	print(smile(mouth))
	cv2.waitKey(0)
cv2.destroyAllWindows()

