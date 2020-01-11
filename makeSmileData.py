import imutils, numpy as np, time, dlib, cv2
from scipy.spatial import distance as dist
from imutils import face_utils;import glob

def makeData(fileList,label):
	for file in fileList:
		img = cv2.imread(file)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		rects = detector(gray, 0)
		if	len(rects) == 1:
			shape = predictor(gray, rects[0])
            #use dlib to detect where the mouth is in the picture
			shape = face_utils.shape_to_np(shape)
			mouth = shape[mStart:mEnd]
			left = min(mouth[:,0]);right=max(mouth[:,0])
			top = min(mouth[:,1]);bottom=max(mouth[:,1])
			width = right-left
			height = bottom-top
			biggest = max(width,height)		
			mouth[:,0] = mouth[:,0]-left
			mouth[:,1] = mouth[:,1]-top
            #scaling the mouth points so all input data have the same dimensions
			for point in range(mouth.shape[0]):
				for coord in range(mouth.shape[1]):
					mouth[point,coord] = int(mouth[point,coord] * 100/biggest)
			cv2.rectangle(img,(left,top),(right,bottom),(255,255,255),2)
			for i in range(mouth.shape[0]-1):
				cv2.line(img,(mouth[i][0],mouth[i][1]),(mouth[i+1][0],mouth[i+1][1]),(255,255,255),1)
            #save all data in a single csv file where the first column contains the label
			print(label+",".join(map(str,mouth.reshape([40]))))
			cv2.imshow('img',img)
			cv2.waitKey(1)

shape_predictor= "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)
cv2.namedWindow('img',cv2.WINDOW_AUTOSIZE)
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
smileFiles=(glob.glob("/Users/faymita29/Desktop/AMLS/AMLS_19-20_SN16015140/Datasets/A2/train/yes/*.jpg"))+(glob.glob("/Users/faymita29/Desktop/AMLS/AMLS_19-20_SN16015140/Datasets/A2/validate/yes/*.jpg"))
noSmileFiles=(glob.glob("/Users/faymita29/Desktop/AMLS/AMLS_19-20_SN16015140/Datasets/A2/train/no/*.jpg"))+(glob.glob("/Users/faymita29/Desktop/AMLS/AMLS_19-20_SN16015140/Datasets/A2/validate/no/*.jpg"))
makeData(smileFiles,"0,")
makeData(noSmileFiles,"1,")
cv2.destroyAllWindows()