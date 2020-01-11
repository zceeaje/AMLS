from sys import platform
dl = ('/','\\')[platform=="win32"]	#Setup folder delimeter for windows or mac/unix
dataFolder = "Datasets" + dl
