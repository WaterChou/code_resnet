import numpy as np
import tensorflow as tf

import singleFilePoccess
import read_test_data

#import singleFilePoccess
import bigbatch

filep = 'C:/Users/fengwusheng1/Desktop/615/test/ScenarioFolder/'
fileq = 'C:/Users/fengwusheng1/Desktop/615/pathfolder1/'
modelPath = "model/model"

def testpath(environment, Spoint):
    #当前点无人机探测到的环境信息
    detectFile = singleFilePoccess.singleDetect(environment, Spoint)
    #神经网络测试输出
    label = testnerualnetwork(detectFile)
    #算出标签转化成偏转角度
    #theta1 = computeAngel(labelFile)
    #加上之前的角度
    #thetaSum = theta1 + thetaSum
    #算出更新点的位置
    #point = [labelFile[0][0]*100, labelFile[0][1]*100]
    #point = labelFile*100
    step = 5
    if label == 1:
        point = [Spoint[0]+step, Spoint[1]]
    elif label == 2:
        point = [Spoint[0]+step, Spoint[1]+step]
    elif label == 3:
        point = [Spoint[0], Spoint[1]+step]
    elif label == 4:
        point = [Spoint[0]-step, Spoint[1]+step]
    elif label == 5:
        point = [Spoint[0]-step, Spoint[1]]
    elif label == 6:
        point = [Spoint[0]-step, Spoint[1]-step]
    elif label == 7:
        point = [Spoint[0], Spoint[1]-step]
    elif label == 8:
        point = [Spoint[0]+step, Spoint[1]-step]
    else:
        print(False)
    
    
    #point = labelFile
    #print(point)
    #返回下一点位置
    return point

#输出DNN预测输出（运动方向）
def testnerualnetwork(testdata1):
    
    x = np.expand_dims(testdata1, axis=0)#扩维的
    #print(x.shape)
    testlabel1 = my_model.predict(x)
    maxaction = 0
    for i in range(len(testlabel1[0])):
        if testlabel1[0][i] > testlabel1[0][maxaction]:
            maxaction = i
    #[[0 1 0]]二维的array
    return maxaction

def predict(sess, env, X_data, batch_size=64):

    print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.empty((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = sess.run(ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    
    print()
    return yval
 


def main():
    
    class Dummy:
        pass

    env = Dummy()
    
    n_classes = 8
    training_flag = True
    fgsm_flag = False
    loadModel_flag = False
    label = 9
    
    fileP = 'D:/PostGraduate/Code/DNN_Pathplanning_adversarial/dataset/test/pathfolderlabel/'
    fileS = 'D:/PostGraduate/Code/DNN_Pathplanning_adversarial/dataset/test/ScenarioFolder/'
    
    detectMap, labelMap = read_test_data.pre(fileS, fileP)
         
    n_detectMap = detectMap.shape[0]
    
    with tf.variable_scope('model',reuse=tf.AUTO_REUSE):
        env.x = tf.placeholder(tf.float32, (None, 32, 32, 2), name = 'x')
        #env.y = tf.placeholder(tf.float32, (None, n_classes), name = 'y')
        #env.is_training = tf.placeholder_with_default(True, (), name = 'is_training')
        env.ybar = resnet50model.ResNet50(env.x, training_flag, n_classes)
      
    saver = tf.train.Saver()
    tf.reset_default_graph()
    
    with tf.Session() as sess:
        
        saver.restore(sess, modelPath)
        
        while label is not 0:
            predLabel[i] = predict(sess, detectMap[i], 1)
        
        
    

if __name__ == "main":
    main()

    