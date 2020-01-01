#================================================
#Script to organize data into subfolders by class
#================================================
import os, shutil, sys
dataFolder = r"/Users/faymita29/Desktop/AMLS/providedDatasets"
dataSets = ['A','B'];tasks = ['A1','A2','B1','B2'];
classesInDataset = {'A':['gender','smiling'],'B':['face_shape','eye_color']}
rootFolders = {	'A': dataFolder + "/celeba",'B': dataFolder + "/cartoon_set"}
categories = {'A1':'gender','A2':'smiling','B1':'face_shape','B2':'eye_color'}
tasksByCategory = {'gender':'A1','smiling':'A2','face_shape':'B1','eye_color':'B2'}
classes = {	'gender':{'-1':'female','1':'male'},
			'smiling':{'-1':'no','1':'yes'},
			'face_shape':{'0':'0','1':'1','2':'2','3':'3','4':'4'},
			'eye_color':{'0':'brown','1':'blue','2':'green','3':'grey','4':'black'}}

#First make the destination folders
destinationFolders = {}

for task in tasks:
		category = categories[task]
		myClasses = classes[category]
		destinationFolders[task] = dataFolder + "/"+task
		os.mkdir(destinationFolders[task])
		for myClass in myClasses.values():
			os.mkdir(destinationFolders[task]+"/"+myClass)

for dataSet in dataSets:
	rootFolder = rootFolders[dataSet]
	photoSourceFolder = rootFolder + "/img"
	labelFileName = rootFolder + "/labels.csv"
	labelFile = open(labelFileName,'r');labelLines = labelFile.readlines();labelFile.close()
	titles = labelLines[0].split();labelLines.pop(0)
	try:
		fileCol = titles.index('img_name')+1
	except:
		fileCol = titles.index('file_name')+1
	class1Col = titles.index(classesInDataset[dataSet][0])
	class2Col = titles.index(classesInDataset[dataSet][1])
	for line in labelLines: 
		lineList = line.split()
		sourceFileName = photoSourceFolder + "/" + lineList[fileCol]
		class1Value = lineList[class1Col+1];class2Value = lineList[class2Col+1]
		class1Category = titles[class1Col];class2Category = titles[class2Col]
		class1ValueName = classes[class1Category][class1Value]
		class2ValueName = classes[class2Category][class2Value]
		task1Name = tasksByCategory[class1Category]
		task2Name = tasksByCategory[class2Category]		
		dest1Folder = dataFolder + "/" + task1Name + "/" + class1ValueName
		dest2Folder = dataFolder + "/" + task2Name + "/" + class2ValueName
#		print(sourceFileName,"---->",dest1Folder,"&&",dest2Folder)
		shutil.copyfile(sourceFileName,dest1Folder+"/"+lineList[fileCol])
		shutil.copyfile(sourceFileName,dest2Folder+"/"+lineList[fileCol])
sys.exit(0)
