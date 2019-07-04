# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 15:00:24 2018

@author: fengwusheng1
"""

import os
import numpy as np
import singleFilePoccess

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
    print(len(sceneList), len(threatList))
    return sceneList, threatList 

def pathFile(filepath):
    pathDir = os.listdir(filepath)
    pathList = []
    for s in pathDir:
        newDir=os.path.join(filepath,s)
        pathList.append(newDir)
    print(len(pathList))
    return pathList


def pre(fileS, fileP):
    
    sceneList, threatList = eachFile(fileS)
    pathList = pathFile(fileP)
#print(len(pathList),len(threatList))
    if len(sceneList) == len(threatList):
        if len(pathList) == len(threatList):
            size = len(pathList)
        else:
            print('False')
    else:
        print('False')
      
    for i in range(size):
        if i == 0:
            detectFile, labelFile = singleFilePoccess.singleScene(sceneList[i], threatList[i], pathList[i])

        else:
            detect, label = singleFilePoccess.singleScene(sceneList[i], threatList[i], pathList[i])
            #path = 
            if np.size(label,0) == 1:
                i=i+1
            else:   
                print(detect.shape)
                print(label.shape)
                detectFile = np.concatenate((detectFile, detect))
                labelFile = np.concatenate((labelFile, label))
                print(i)
                
    return detectFile,labelFile
