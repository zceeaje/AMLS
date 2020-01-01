#================================================
#Script to organize data into subfolders by class
#================================================
import os, shutil, sys;import os.path;from os import path;import random
inputFolder = r"/Users/faymita29/Desktop/AMLS/providedDatasets"
outputFolder = r"/Users/faymita29/Desktop/AMLS/AMLS_19-20_SN16015140/Datasets"
dataSets = ['A','B'];tasks = ['A1','A2','B1','B2']
imageCols = {'A':'img_name','B':'file_name'}
classesInDataset = {'A':['gender','smiling'],'B':['face_shape','eye_color']}
labelFolders = {'A': inputFolder + "/celeba",'B': inputFolder + "/cartoon_set"}
imageFolders = {'A': labelFolders['A'] + "/img",'B': labelFolders['B'] + "/img"}
photoSourceFolders = {'A': labelFolders['A'] + "/img", 'B': labelFolders['B'] + "/img"}
categories = {'A1':'gender','A2':'smiling','B1':'face_shape','B2':'eye_color'}
tasksByCategory = {'gender':'A1','smiling':'A2','face_shape':'B1','eye_color':'B2'}
classes = {	'gender':{'-1':'female','1':'male'},'smiling':{'-1':'no','1':'yes'},
			'face_shape':{'0':'0','1':'1','2':'2','3':'3','4':'4'},
			'eye_color':{'0':'brown','1':'blue','2':'green','3':'grey','4':'black'}}
trainPart,validatePart,testPart = 0.6,0.2,0.2

#===========First, make the folder structure=================
destinationFolders = {}
def makeFolder(folderName):
	if not(path.isdir(folderName)): os.mkdir(folderName)
listOfFiles = {}
makeFolder(outputFolder)
for task in tasks:
		listOfFiles[task] = {}
		category = categories[task]
		myClasses = classes[category]
		destinationFolders[task] = outputFolder + "/"+task;
		makeFolder(destinationFolders[task])
		for set in ["train","validate","test"]: 
			setFolder = destinationFolders[task]+"/"+set
			makeFolder(setFolder)
			for myClass in myClasses.values():
				makeFolder(setFolder+"/"+myClass)
				listOfFiles[task][myClass] = []

#===========Second,get a list of all the files=================
for dataSet in dataSets:
	labelFile = open(labelFolders[dataSet] + "/labels.csv",'r');
	labelLines = labelFile.readlines();labelFile.close()
	titles = labelLines.pop(0).split()
	fileCol = titles.index(imageCols[dataSet])+1
	for line in labelLines: 
		lineList = line.split()
		sourceFileName = photoSourceFolders[dataSet] + "/" + lineList[fileCol]
		for taskNum in range(2):
			classCol = titles.index(classesInDataset[dataSet][taskNum])
			classValue = lineList[classCol+1];
			classCategory = titles[classCol];
			classValueName = classes[classCategory][classValue]
			taskName = tasksByCategory[classCategory]
			listOfFiles[taskName][classValueName].append(lineList[fileCol])

#===========Finally, put the files in the appropriate folders============
for task in tasks:
	category = categories[task]
	for myClass in classes[category].keys():
		value = classes[category][myClass]
		random.shuffle(listOfFiles[task][value])
		sizes = {};	sets = {}
		sizes["train"] = int(trainPart*len(listOfFiles[task][value]))
		sizes["validate"] = int(validatePart*len(listOfFiles[task][value]))
		sizes["test"] = len(listOfFiles[task][value])-sizes["train"]-sizes["validate"]
		sets["train"] = listOfFiles[task][value][:sizes["train"]]
		sets["validate"] = listOfFiles[task][value][sizes["train"]:(sizes["train"]+sizes["validate"])]
		sets["test"] = listOfFiles[task][value][(sizes["train"]+sizes["validate"]):]
		sourceFolder = photoSourceFolders[task[0]] + "/"
		for mySet in ["train","validate","test"]:
			destFolder = destinationFolders[task] + "/" + mySet + "/" + value + "/"
			print("Copying %d files from %s to %s"%(len(sets[mySet]),sourceFolder,destFolder))
			for myFile in sets[mySet]:
				shutil.copyfile(sourceFolder+myFile,destFolder+myFile)
