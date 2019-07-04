import os
import numpy as np
import singleFilePoccess

#分类场景文件和威胁文件
def eachFile(filepath):
    
    pathDir = os.listdir(filepath)      #获取当前路径下的文件名，返回List
    sceneList = []
    threatList = []
    for s in pathDir:
        newDir=os.path.join(filepath,s)
        #将文件命加入到当前文件路径后面
        #print(newDir)
        if os.path.isfile(newDir) :         #如果是文件
            if os.path.splitext(newDir)[1]==".txt":  #判断是否是txt
                #sceneFileName, threatFileName = readFile(newDir)
                if 'B' in newDir:
                    threatList.append(newDir)
                else:
                    sceneList.append(newDir)                              
        else:
            eachFile(newDir)                #如果不是文件，递归这个文件夹的路径
    
    print("len of sceneList, threatList")
    print(len(sceneList), len(threatList))
    
    return sceneList, threatList 

def pathFile(filepath):
    
    pathDir = os.listdir(filepath)  #返回指定的文件夹包含的文件或文件夹的名字的列表
    pathList = []
    
    for s in pathDir:
        
        newDir=os.path.join(filepath,s)
        pathList.append(newDir)
    
    print("len of pathList: %d" %len(pathList))
    
    return pathList
            
#def readFile(fileName):
#    scenefileName = []
#    threatfileName = []
#    if 'A' in fileName:
#        scenefileName = fileName
#    elif 'B' in fileName:
#        threatfileName = fileName
#    return scenefileName, threatfileName

def pre(fileS, fileP):
 
    #sceneList,threatList: 场景文件列表和威胁度文件列表
    #pathList：路径标签文件列表
    sceneList, threatList = eachFile(fileS)
    pathList = pathFile(fileP)

    if len(sceneList) == len(threatList):
        if len(pathList) == len(threatList):
            size = len(pathList)
            #print('True')
        else:
            print('False, len(pathList) != len(threatList)')
    else:
        print('False, len(sceneList) != len(threatList)')
    
    for i in range(size):
                
        if i == 0:
            detectFile, labelFile = singleFilePoccess.singleScene(sceneList[i], threatList[i], pathList[i])
            print(sceneList[i])

        else:
            detect, label = singleFilePoccess.singleScene(sceneList[i], threatList[i], pathList[i])
            print(sceneList[i])
            if np.size(label,0) == 1:
                i = i
            else: 
                print("i: %d" %i)
                
                print("detect shape: ") 
                print(detect.shape)
                
                print("label shape: ") 
                print(label.shape)
                
                detectFile = np.concatenate((detectFile, detect))
                labelFile = np.concatenate((labelFile, label))
                
    return detectFile, labelFile


