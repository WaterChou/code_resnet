
#路径规划数据预处理
import numpy as np

# 对距离矩阵进行归一化处理
def zeroOneNormailzation(input):
    # input (32,32,2)
    temp = np.zeros(shape=(input.shape[0], input.shape[1]))
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            temp[i][j] = input[i][j][0]

    max = np.nanmax(temp)
    min = np.nanmin(temp)
    # print('max=' + str(max) + ',min=' + str(min))

    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            # print('before: '+str(input[i][j][0]))
            input[i][j][0] = (input[i][j][0]-min)/(max-min)
            # print('after: ' + str(input[i][j][0]))

    return input

#拼接整个地图的距离和威胁矩阵
def mergeEnvironment(scene, threat):
    sizescene = scene.shape  # (101,101)
    #sizethreat = threat.shape
    # scene /= 71 #归一化
    environment = np.zeros([sizescene[0], sizescene[1], 2])
    for i in range(sizescene[0]):
        for j in range(sizescene[1]):
            environment[i][j] = [scene[j][i], threat[j][i]]

    return environment

#无人机在当前航路点探测信息
def singleDetect(environment, pathPoint1):
    circleR = 16    # 探测矩阵为32*32
    detectFile = np.zeros([2*circleR, 2*circleR, 2])
    for i in range(2*circleR):
        for j in range(2*circleR):
            if int(pathPoint1[0] + i -circleR) >=0 and int(pathPoint1[0] + i -circleR) <=100 and int(pathPoint1[1] + j -circleR) >= 0 and int(pathPoint1[1] + j -circleR) <= 100:
                detectFile[i][j] = environment[int(pathPoint1[0] + i -circleR)][int(pathPoint1[1] + j -circleR)]
            else:
                detectFile[i][j] = np.array([1, 1])
    detectFile = zeroOneNormailzation(detectFile)
    return detectFile

#无人机在当前场景下，整条路径的探测信息合集
def pathDetect(pathPoint, environment):
    num = pathPoint.shape
    #num[0]
    for i in range(num[0]):
        if i == 0:
            detectSum = singleDetect(environment, pathPoint[i])
        elif i == 1:
            d = singleDetect(environment, pathPoint[i])
            detectSum = np.vstack(([detectSum], [d]))
        elif i >= 2:
            d = singleDetect(environment, pathPoint[i])
            detectSum = np.concatenate((detectSum, [d]))

    return detectSum

'''
#无人机路径信息分割 
#10 = 2(位置坐标x,y)+8(8个方位角的概率)
def dividePath(path):
    a = path.reshape(-1,10)
    
    print("path shape:")
    print(a.shape)
    
    pathPoint = a[:,:2]
    pathLabel = a[:,2:]
    #pathPointLabel = path[1:,:2]*0.01
    pathPointLabel = a[:,:2]
    
    return pathPoint, pathLabel, pathPointLabel
'''

#生成路径点和方向标签
#step：A星算法步长
def generatePathPointLable(pathfile, step):
    
    file = open(pathfile , "r")
    
    lines = file.readlines()
    lines = lines[1:]   #文件第一行不是路径点信息，舍弃
    
    n_lines = len(lines)
    
    pathPoint = np.zeros((n_lines,2))
    pathPointLabel = np.zeros((n_lines,8))
    
    for i in range(n_lines):
        pathPointStr = lines[i].split(' ')

        pathPoint[i][0] = float(pathPointStr[0])#x
        pathPoint[i][1] = float(pathPointStr[1])#y
    
    for i in range(n_lines-1):#生成运动方向标签矩阵n_line*8
        
        if (pathPoint[i+1][0] == pathPoint[i][0] + step) and (pathPoint[i+1][1] == pathPoint[i][1]):
            pathPointLabel[i][0] = 1
            
        elif (pathPoint[i+1][0] == pathPoint[i][0] + step) and (pathPoint[i+1][1] == pathPoint[i][1] - step):
            pathPointLabel[i][1] = 1
            
        elif (pathPoint[i+1][0] == pathPoint[i][0]) and (pathPoint[i+1][1] == pathPoint[i][1] - step):
            pathPointLabel[i][2] = 1
            
        elif (pathPoint[i+1][0] == pathPoint[i][0] - step) and (pathPoint[i+1][1] == pathPoint[i][1] - step):
            pathPointLabel[i][3] = 1
            
        elif (pathPoint[i+1][0] == pathPoint[i][0] - step) and (pathPoint[i+1][1] == pathPoint[i][1]):
            pathPointLabel[i][4] = 1
            
        elif (pathPoint[i+1][0] == pathPoint[i][0] - step) and (pathPoint[i+1][1] == pathPoint[i][1] + step):
            pathPointLabel[i][5] = 1
            
        elif (pathPoint[i+1][0] == pathPoint[i][0]) and (pathPoint[i+1][1] == pathPoint[i][1] + step):
            pathPointLabel[i][6] = 1
            
        else:
            pathPointLabel[i][7] = 1
        
    
    return pathPoint, pathPointLabel
            

#def singleLabel(pathlabels):
#    labelFile = pathlabels
#    return labelFile

    
#主函数
def singleScene(scenePath, threatPath, pathPath):
    
    scene = np.loadtxt(scenePath)
    
    threat = np.loadtxt(threatPath)
    
    environment = mergeEnvironment(scene, threat)
    
    #pathPoint, labelMat, pathPointLabel = dividePath(path)
    pathPoint, labelMat = generatePathPointLable(pathPath, step=1)   #pathPoint是路径的坐标点， labelMat是运动方向标志
    
    detectMat = pathDetect(pathPoint, environment)       
    
    return detectMat, labelMat

#detectFile, labelFile = singleScene(scenePath, threatPath, pathPath)