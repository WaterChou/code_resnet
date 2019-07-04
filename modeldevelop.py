# -*- coding: utf-8 -*-
"""
Created on Fri May 10 08:58:45 2019

@author: 49435
"""
import os
import tensorflow as tf
import numpy as np
import re
import xlwt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import resnet50model
import read_train_data
import read_test_data
import singleFilePoccess

from attacks import fgm

# 指定0号GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train(sess, writer, env, X_data, y_data, X_valid=None, y_valid=None,
          epochs=1, load=False, shuffle=True, batch_size=64, name='model'):

    if load:
        if not hasattr(env, 'saver'):
            return print('\nError: cannot find saver op')
        print('\nLoading saved model')
        return env.saver.restore(sess, './model/{}'.format(name))

    print('\nTrain model')
    loss, acc = 0, 0
    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    valid_result = np.zeros(shape=(epochs, 2))

    for epoch in range(epochs):

        print('\nEpoch {0}/{1}'.format(epoch + 1, epochs))

        if shuffle:
            print('\nShuffling data')
            ind = np.arange(n_sample)  # 生成长度为n_sample的等差数列
            np.random.shuffle(ind)  # 随机打乱ind内数据顺序
            X_data = X_data[ind]
            y_data = y_data[ind]
            # counter = np.zeros(shape=(3,))

        for batch in range(n_batch):
            print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
            start = batch * batch_size
            end = min(n_sample, start + batch_size)

            if np.isnan(X_data[start:end]).any():   # whether any element is True
                print('nan data in batch: '+str(batch))
                return


            sess.run(env.train_op, feed_dict={env.x: X_data[start:end], env.y: y_data[start:end], env.train_flag: True})
            batch_loss, batch_acc = sess.run([env.loss, env.acc],
                                             feed_dict={env.x: X_data[start:end], env.y: y_data[start:end],
                                                        env.train_flag: True})

            if np.isnan(batch_loss):
                print('batch_loss is nan')

            loss += batch_loss * (end-start)
            acc += batch_acc * (end-start)

        loss /= n_sample
        acc /= n_sample
        print('\nEvaluate on  train')
        print(' loss: {0:.5f} acc: {1:.5f}'.format(loss, acc))

        if X_valid is not None:
            print('\nEvaluate on valid')
            # evaluate(sess, env, X_valid, y_valid)
            loss, acc = evaluate(sess, env, X_valid, y_valid)
            valid_result[epoch][0] = loss
            valid_result[epoch][1] = acc

    if hasattr(env, 'saver'):
        print('\nSaving model')
        os.makedirs('model_epoch', exist_ok=True)
        env.saver.save(sess, 'model_epoch/{}'.format(name+str(epoch)))

    writer.train.close()
    writer.valid.close()

    if hasattr(env, 'saver'):
        print('\nSaving model')
        os.makedirs('model', exist_ok=True)
        env.saver.save(sess, 'model/{}'.format(name))

    if X_valid is not None:
        return valid_result


def evaluate(sess, env, X_data, y_data, batch_size=64):

    print('\nEvaluating')

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    loss, acc = 0, 0

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        cnt = end - start
        batch_loss, batch_acc = sess.run([env.loss, env.acc],
                                         feed_dict={env.x: X_data[start:end], env.y: y_data[start:end],
                                                    env.train_flag: False})
        loss += batch_loss * cnt
        acc += batch_acc * cnt
    loss /= n_sample
    acc /= n_sample

    print(' loss: {0:.5f} acc: {1:.5f}'.format(loss, acc))
    return loss, acc      


def predict(sess, env, X_data, batch_size=64):

    print('\nPredicting')
    n_classes = env.ybar.get_shape().as_list()[1]

    n_sample = X_data.shape[0]
    n_batch = int((n_sample+batch_size-1) / batch_size)
    yval = np.zeros((n_sample, n_classes))

    for batch in range(n_batch):
        print(' batch {0}/{1}'.format(batch + 1, n_batch), end='\r')
        start = batch * batch_size
        end = min(n_sample, start + batch_size)
        y_batch = sess.run(env.ybar, feed_dict={env.x: X_data[start:end]})
        yval[start:end] = y_batch
    print()
    return yval


def split_data(x_data, y_data):

    ind = np.arange(x_data.shape[0])
    np.random.shuffle(ind)
    x_data = x_data[ind]
    y_data = y_data[ind]
    
    n = int(x_data.shape[0] * 0.8)
    
    x_valid = x_data[n:]
    x_train = x_data[:n]
    y_valid = y_data[n:]
    y_train = y_data[:n]

    n = int(x_valid.shape[0] * 0.5)
    x_test = x_valid[n:]
    x_valid = x_valid[:n]
    y_test = y_valid[n:]
    y_valid = y_valid[:n]
    
    return x_test, x_valid, x_train, y_test,  y_valid, y_train


# 输出DNN预测输出（运动方向）
def testnerualnetwork(sess,env,testdata1):
    
    x = np.expand_dims(testdata1, axis=0)#扩维的
    #print(x.shape)
    testlabel1 = predict(sess, env, x, batch_size=64)
    maxaction = 0
    for i in range(len(testlabel1[0])):
        if testlabel1[0][i] > testlabel1[0][maxaction]:
            maxaction = i
    #[[0 1 0]]二维的array
    return maxaction


def testpath(sess, env, environment, step,Spoint):
    #当前点无人机探测到的环境信息
    detectMap = singleFilePoccess.singleDetect(environment, Spoint)
    #神经网络测试输出
    label = testnerualnetwork(sess, env, detectMap)
    #算出更新点的位置
    #step = 1
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
    
    #返回下一点位置
    return point


def is_arriving(nowPoint, endPoint):
    
     if np.sqrt(np.sum(np.square(nowPoint-endPoint))) > 7.5:#未到终点附近
         return False
     else:
         return True


def is_inScene(nowPoint, width, high):
    
    if nowPoint[0]>=0 and nowPoint[0]<=width and nowPoint[1]>=0 and nowPoint[1]<=high:
        return True
    else:
        return False


class Dummy:
    pass


env = Dummy()
writer = Dummy()

n_classes = 8
n_predict = 5
scene_width = 100
scene_high = 100
step = 1
loadData_flag = True


fileP = 'dataset/train/PathFolder/'
fileS = 'dataset/train/ScenarioFolder/'
'''
fileP = 'dataset/simple scene/PathFolder/'
fileS = 'dataset/simple scene/ScenarioFolder/'
'''
# predPathPoint_path = 'res/predPathPoint.xls'

X_train_after_processing_path = 'after_processing/simple scene/X_train.npy'
Y_train_after_processing_path = 'after_processing/simple scene/Y_train.npy'
X_valid_after_processing_path = 'after_processing/simple scene/X_valid.npy'
Y_valid_after_processing_path = 'after_processing/simple scene/Y_valid.npy'
X_test_after_processing_path = 'after_processing/simple scene/X_test.npy'
Y_test_after_processing_path = 'after_processing/simple scene/Y_test.npy'
valid_result_path = 'valid_result.npy'
'''
X_train_after_processing_path = 'after_processing/X_train.npy'
Y_train_after_processing_path = 'after_processing/Y_train.npy'
X_valid_after_processing_path = 'after_processing/X_valid.npy'
Y_valid_after_processing_path = 'after_processing/Y_valid.npy'
X_test_after_processing_path = 'after_processing/X_test.npy'
Y_test_after_processing_path = 'after_processing/Y_test.npy'
'''

with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
    env.x = tf.placeholder(tf.float32, (None, 32, 32, 2), name='x')
    env.y = tf.placeholder(tf.float32, (None, n_classes), name='y')
    env.train_flag = tf.placeholder_with_default(False, (), name='train_flag')
    env.ybar, env.logits = resnet50model.ResNet50(env.x, is_training=env.train_flag, classes=n_classes, logits=True)

    with tf.variable_scope('acc'):
        count = tf.equal(tf.argmax(env.y, axis=1), tf.argmax(env.ybar, axis=1))
        env.acc = tf.reduce_mean(tf.cast(count, tf.float32), name='acc')

    with tf.variable_scope('loss'):
        ce = tf.nn.softmax_cross_entropy_with_logits(labels=env.y, logits=env.logits)
        # ce = tf.keras.losses.categorical_crossentropy(y_true=env.y, y_pred=env.ybar)
        env.loss = tf.reduce_mean(ce, name='loss')

    with tf.variable_scope('train_op'):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)  # defalut learning_rate=0.001
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)     # 保证train_op在update_ops执行之后再执行。
        with tf.control_dependencies(update_ops):
            env.train_op = optimizer.minimize(env.loss)     # necessary for batch normalization

    # env.conv1_w = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    env.saver = tf.train.Saver()

#    #生成对抗样本时需要增加的计算图
#    with tf.variable_scope('model', reuse=True):
#        env.fgsm_eps = tf.placeholder(tf.float32, (), name='fgsm_eps')
#        env.fgsm_epochs = tf.placeholder(tf.int32, (), name='fgsm_epochs')
#        env.x_fgm = fgm(resnet50model.ResNet50, env.x, is_training=training_flag, 
#                        classes=n_classes, sign = fgsm_flag, epochs=env.fgsm_epochs, eps=env.fgsm_eps) 
#    
#   load data
if loadData_flag is False:
    X_train, Y_train = read_train_data.pre(fileS, fileP)
    X_test, X_valid, X_train, Y_test, Y_valid, Y_train = split_data(X_train, Y_train)
    print('\ntrain_set_shape')
    print(np.shape(X_train))
    print(np.shape(Y_train))
    print('\nvalid_set_shape')
    print(np.shape(X_valid))
    print(np.shape(Y_valid))
    print('\ntest_set_shape')
    print(np.shape(X_test))
    print(np.shape(Y_test))

    os.makedirs('after_processing/simple scene', exist_ok=True)
    np.save(X_train_after_processing_path, X_train)
    np.save(Y_train_after_processing_path, Y_train)
    np.save(X_valid_after_processing_path, X_valid)
    np.save(Y_valid_after_processing_path, Y_valid)
    np.save(X_test_after_processing_path, X_test)
    np.save(Y_test_after_processing_path, Y_test)

else:   # 加载处理好的数据
    X_train = np.reshape(np.load(X_train_after_processing_path), [-1, 32, 32, 2])
    Y_train = np.reshape(np.load(Y_train_after_processing_path), [-1, 8])
    X_valid = np.reshape(np.load(X_valid_after_processing_path), [-1, 32, 32, 2])
    Y_valid = np.reshape(np.load(Y_valid_after_processing_path), [-1, 8])
    X_test = np.reshape(np.load(X_test_after_processing_path), [-1, 32, 32, 2])
    Y_test = np.reshape(np.load(Y_test_after_processing_path), [-1, 8])
    print('\ntrain_set_shape')
    print(np.shape(X_train))
    print(np.shape(Y_train))
    print('\nvalid_set_shape')
    print(np.shape(X_valid))
    print(np.shape(Y_valid))
    print('\ntest_set_shape')
    print(np.shape(X_test))
    print(np.shape(Y_test))


print('Initializing graph \n')
graph = tf.get_default_graph()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    writer.train = tf.summary.FileWriter('logs/train/', sess.graph)
    writer.valid = tf.summary.FileWriter('logs/valid/')

    # fgsm_flag = False
    loadModel_flag = False

    print('\nTraining')
    train(sess, writer, env, X_train, Y_train, X_valid=X_valid, y_valid=Y_valid,
          load=loadModel_flag, shuffle=True, epochs=5, batch_size=64, name='model')

    print('\nEvaluating on test data')
    evaluate(sess, env, X_test, Y_test)

    # np.save(valid_result_path, valid_result)


    '''        
    if loadModel_flag is False:
        
        print('\nTraining')
        trianing_flag = True
        train_loss_list, valid_loss_list = train(sess, env, X_train, Y_train, X_valid, Y_valid, 
                                                 load = loadModel_flag, epochs=25, batch_size=64, name='model')
        
        print('\nEvaluating on test data') 
        trianing_flag = False
        evaluate(sess, env, X_test, Y_test)
    
    else:
        
        print('\nPredicting')    
        #载入模型
        train(sess, env, X_train, Y_train, X_valid, Y_valid, load=loadModel_flag, epochs=1, name='model')
        
        file_predPathPoint = xlwt.Workbook(encoding='utf-8', style_compression = 0)            
        
        startEndPoint = np.empty((n_predict, 2, 2))#保存起点和终点
        
        pathFileList = read_test_data.pathFile(filePTest)
        pathFileList = pathFileList[:, n_predict]#绘制前n_predict场景的路径
        sceneFileList , threatFileList = read_test_data.eachFile(fileSTest)
        sceneFileList , threatFileList = sceneFileList[:, n_predict], threatFileList[:, n_predict]
        end_flag = False#是否到达终点
        
        for i in range(n_predict):
            
            sheet = file_predPathPoint.add_sheet(obj = str(i), cell_overwrite_ok = True)
            
            sceneTemp, threatTemp = sceneFileList[i], threatFileList[i]
            environmentTemp = singleFilePoccess.mergeEnvironment(sceneTemp, threatTemp)
            pathPointTemp, _ = singleFilePoccess.generatePathPointLable(pathFileList[i], step)   #某个路径文件中所有的路径点
            
            startEndPoint[i][0]= pathPointTemp[0]   #起点
            startEndPoint[i][1] = pathPointTemp[-1] #终点
            sheet.write(j, 0, startEndPoint[i][0])
            sheet.write(j, 1, startEndPoint[i][1])
            
            j = 0
            predPointPath = np.copy(np.expand_dims(startEndPoint[i], axis=0))
            while end_flag is not True:
                if j == 0:                      
                    nextPoint = testpath(sess, env, environmentTemp, startEndPoint[i],step)
                else:
                    if is_arriving(nextPoint,startEndPoint[i][-1]) is  False:#未到达终点
                        if is_inScene(nextPoint, scene_width, scene_high) is True:
                            nextPoint = testpath(sess, env, environmentTemp, nextPoint,step)
                        else:
                            j = j-1
                    else:
                        end_flag = True
                predPointPath = np.vstack((predPointPath, nextPoint))
                j = j+1      
                sheet.write(j, 0, nextPoint)
                sheet.write(j, 1, nextPoint)
                
            file_predPathPoint.save(predPathPoint_path)
            
            #画图
            #画场景
            plt.axis([0, scene_width, 0, scene_high])
            for p in range(environmentTemp.shape[0]):
                for q in range(environmentTemp.shape[1]):
                    if environmentTemp[p][q][1] >= 0.85:#威胁度大于0.85才显示为障碍物
                        plt.scatter(p, q, c='b', alpha=environmentTemp[p][q][1])
            
            #画路径
            n_pointPath = predPointPath.shape[0]
            x = []
            y = []
           
            for k in range(n_pointPath):
                x.append(predPointPath[k][0])
                y.append(predPointPath[k][1])
            
            plt.plot(x, y, "r-", marker='o', linewidth = 1)
            plt.plot(pathPointTemp[:,0], pathPointTemp[:,1], "g--", marker='o', linewidth = 1)
            
            plt.savefig('./res/'+'%04d'%i+'.png')
            plt.close('all')
    '''
