import imutils, numpy as np, time, dlib, cv2
from scipy.spatial import distance as dist
from imutils import face_utils;import glob, sys
rootFolder = "c:\\Users\\daron\\Documents\\AMLS\\newData\\B1\\"

def makeData(fileList,label):
	for file in fileList:
		sys.stderr.write(".");sys.stderr.flush()		
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
			for i in range(jaw.shape[0]-1):
				cv2.line(img,(jaw[i][0],jaw[i][1]),(jaw[i+1][0],jaw[i+1][1]),(255,0,0),1)
			print(label+","+",".join(map(str,jaw.reshape([34]))))
			cv2.imshow('img',img)
			cv2.waitKey(1)

shape_predictor= "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
cv2.namedWindow('img',cv2.WINDOW_AUTOSIZE)
for faceShape in range(5):
	fileList= glob.glob(rootFolder + "train\\" + str(faceShape)+"\\*.png") + glob.glob(rootFolder + "validate\\"+str(faceShape)+"\\*.png")
	sys.stderr.write("Processing files for faceShape %d (%d items)\n"%(faceShape, len(fileList)))
	#print(fileList)
	makeData(fileList,str(faceShape))
cv2.destroyAllWindows()