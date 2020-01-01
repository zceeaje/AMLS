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
classes = {	'gender':{'-1':'female','1':'male'},
			'smiling':{'-1':'no','1':'yes'},
			'face_shape':{'0':'0','1':'1','2':'2','3':'3','4':'4'},
			'eye_color':{'0':'brown','1':'blue','2':'green','3':'grey','4':'black'}}
Ntrain,Nvalidate,Ntest = 0.6,0.2,0.2

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
		#listOfFiles[task][category] = {}
		for set in ["train","validate","test"]: 
			setFolder = destinationFolders[task]+"/"+set
			makeFolder(setFolder)
			for myClass in myClasses.values():
				makeFolder(setFolder+"/"+myClass)
				listOfFiles[task][myClass] = []

for dataSet in dataSets:
	labelFileName = labelFolders[dataSet] + "/labels.csv"
	labelFile = open(labelFileName,'r');labelLines = labelFile.readlines();labelFile.close()
	titles = labelLines[0].split();labelLines.pop(0)
	fileCol = titles.index(imageCols[dataSet])+1
	class1Col = titles.index(classesInDataset[dataSet][0])
	class2Col = titles.index(classesInDataset[dataSet][1])
	for line in labelLines: 
		lineList = line.split()
		sourceFileName = photoSourceFolders[dataSet] + "/" + lineList[fileCol]
		class1Value = lineList[class1Col+1];class2Value = lineList[class2Col+1]
		class1Category = titles[class1Col];class2Category = titles[class2Col]
		class1ValueName = classes[class1Category][class1Value]
		class2ValueName = classes[class2Category][class2Value]
		task1Name = tasksByCategory[class1Category]
		task2Name = tasksByCategory[class2Category]		
		#dest1Folder = outputFolder + "\\" + task1Name + "\\" + class1ValueName
		#dest2Folder = outputFolder + "\\" + task2Name + "\\" + class2ValueName
#		listOfFiles[task1Name][class1ValueName].append(sourceFileName)
#		listOfFiles[task2Name][class2ValueName].append(sourceFileName)
		listOfFiles[task1Name][class1ValueName].append(lineList[fileCol])
		listOfFiles[task2Name][class2ValueName].append(lineList[fileCol])
		#shutil.copyfile(sourceFileName,dest1Folder+"\\"+lineList[fileCol])
		#shutil.copyfile(sourceFileName,dest2Folder+"\\"+lineList[fileCol])
#print(listOfFiles)

for task in tasks:
	category = categories[task]
	for myClass in classes[category].keys():
		value = classes[category][myClass]
		random.shuffle(listOfFiles[task][value])
		sizes = {}
		sizes["train"] = int(Ntrain*len(listOfFiles[task][value]))
		sizes["validate"] = int(Nvalidate*len(listOfFiles[task][value]))
		sizes["test"] = len(listOfFiles[task][value])-sizes["train"]-sizes["validate"]
		sets = {}
		sets["train"] = listOfFiles[task][value][:sizes["train"]]
		sets["validate"] = listOfFiles[task][value][sizes["train"]:(sizes["train"]+sizes["validate"])]
		sets["test"] = listOfFiles[task][value][(sizes["train"]+sizes["validate"]):]
		sourceFolder = photoSourceFolders[task[0]] + "/"
		for mySet in ["train","validate","test"]:
			destFolder = outputFolder + "/" + task + "/" + mySet + "/" + value + "/"
			print("Copying %d files from %s to %s"%(len(sets[mySet]),sourceFolder,destFolder))
			for myFile in sets[mySet]:
				shutil.copyfile(sourceFolder+myFile,destFolder+myFile)
