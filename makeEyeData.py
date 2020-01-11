import glob, cv2, math
from statistics import mean, mode
from collections import Counter
leftEyeCentreX = 205; leftEyeCentreY = 260
leftEyeCentre = (leftEyeCentreX,leftEyeCentreY)
pupilRadius = 7; irisRadius = 15
RED = (0,0,255);BLUE=(255,0,0)
fileNum = 0;trainingData=""
labels = {"black":"0","brown":"1","blue":"2","green":"3","grey":"4"}
for use in ["train","validate"]:
	for myClass in ["black","brown","blue","green","grey"]:
		celebFiles=(glob.glob("/Users/faymita29/Desktop/AMLS/AMLS_19-20_SN16015140/Datasets/B2/"+use+"/"+myClass+"/*.png"))
		for file in celebFiles:
			img = cv2.imread(file)
			fileNum +=1;
			#print("Processing file %d: %s..."%(fileNum,file))
			fileStr = labels[myClass]
            #manually find the eyes location and extract it as a donut shape
			for x in range(-irisRadius,irisRadius):
				for y in range(-irisRadius,irisRadius):
					distance = math.sqrt((x*x)+(y*y))
					if (distance > pupilRadius) and (distance < irisRadius):
						COL = img[leftEyeCentreY+y][leftEyeCentreX+x]
                        #save as a single csv file where first column is the label
						fileStr = fileStr + "," + ",".join(map(str,COL))
			print(fileStr)
cv2.destroyAllWindows()
