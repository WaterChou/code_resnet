import tensorflow as tf

_BATCH_NORM_DECAY = 0.99   # Momentum for moving average. 0.997
_BATCH_NORM_EPSILON = 1e-5  # epsilon, added to variance to avoid dividing by zero. 1e-5
_LEAKY_ALPHA = 0

'''
Shortcut为identity mapping
X : input
filters : 1*3 卷积核数目
f: kernel_size = (f,f)
stage : resnet paramater
block : 'a' or 'b' or 'c'
is_training: batch_normalization paramater, 训练时为True,测试时为False
'''


def identity_block(X, f, filters, stage, block, is_training=True):

    res_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + 'branch'

    F1 = filters[0]
    F2 = filters[1]
    F3 = filters[2]
    
    X_shortcut = X
    
    with tf.variable_scope('identity_block_' + str(stage) + block):
        # First component of main path
        X = tf.layers.conv2d(X, filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                             name=res_name_base + '2a',
                             kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
                             #kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
        X = tf.layers.batch_normalization(X, training=is_training, momentum=_BATCH_NORM_DECAY,
                                          epsilon=_BATCH_NORM_EPSILON, name=bn_name_base+'2a')
        X = tf.keras.layers.LeakyReLU(_LEAKY_ALPHA)(X)
        
        # Second component of main path
        X = tf.layers.conv2d(X, filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
                             name=res_name_base + '2b',
                             kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
                             # kernel_initializer=tf.keras.initializers.glorot_normal(seed=0))
        X = tf.layers.batch_normalization(X, training=is_training, momentum=_BATCH_NORM_DECAY,
                                          epsilon=_BATCH_NORM_EPSILON, name=bn_name_base+'2b')
        X = tf.keras.layers.LeakyReLU(_LEAKY_ALPHA)(X)
        
        # Third component of main path
        X = tf.layers.conv2d(X, filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                             name=res_name_base + '2c',
                             kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
        X = tf.layers.batch_normalization(X, training=is_training, momentum=_BATCH_NORM_DECAY,
                                          epsilon=_BATCH_NORM_EPSILON, name=bn_name_base+'2c')
        
        # Final
        X = tf.add(X, X_shortcut)
        X = tf.keras.layers.LeakyReLU(_LEAKY_ALPHA)(X)
        
        return X


'''
Shortcut为convolutional mapping
X : input
filters : 1*3 卷积核数目
f: kernel_size = (f,f)
stage : resnet paramater
block : 'a' or 'b' or 'c'
is_training: batch_normalization paramater, 训练时为True,测试时为False
s: strides = (s,s)
'''
def convolutional_block(X, f, filters, stage, block, is_training=True, s = 2):
    
    res_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + 'branch'
    
    F1 = filters[0]
    F2 = filters[1]
    F3 = filters[2]

    X_shortcut = X
    
    with tf.variable_scope('convolutional_block_' + str(stage) + block):
        # MAIN PATH
        X = tf.layers.conv2d(X, filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid',
                             name=res_name_base + '2a',
                             kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
        X = tf.layers.batch_normalization(X, training=is_training, momentum=_BATCH_NORM_DECAY,
                                          epsilon=_BATCH_NORM_EPSILON, name=bn_name_base+'2a')
        X = tf.keras.layers.LeakyReLU(_LEAKY_ALPHA)(X)
        
        # Second component of main path
        X = tf.layers.conv2d(X, filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
                             name=res_name_base + '2b',
                             kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
        X = tf.layers.batch_normalization(X, training=is_training, momentum=_BATCH_NORM_DECAY,
                                          epsilon=_BATCH_NORM_EPSILON, name=bn_name_base+'2b')
        X = tf.keras.layers.LeakyReLU(_LEAKY_ALPHA)(X)
        
        # Third component of main path
        X = tf.layers.conv2d(X, filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                             name=res_name_base + '2c',
                             kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
        X = tf.layers.batch_normalization(X, training=is_training, momentum=_BATCH_NORM_DECAY,
                                          epsilon=_BATCH_NORM_EPSILON, name=bn_name_base+'2c')
        
        # SHORTCUT PATH
        X_shortcut = tf.layers.conv2d(X_shortcut, filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid',
                                      name=res_name_base + '1',
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0))
        X_shortcut = tf.layers.batch_normalization(X_shortcut, training=is_training, momentum=_BATCH_NORM_DECAY,
                                                   epsilon=_BATCH_NORM_EPSILON, name=bn_name_base+'1')
        
        # Final
        X = tf.add(X, X_shortcut)
        X = tf.keras.layers.LeakyReLU(_LEAKY_ALPHA)(X)
        
        return X


'''
搭建ResNet50模型
'''
def ResNet50(X, is_training=True, classes = 8, logits=False):
  
    #X_input = tf.keras.layers.Input(shape=input_shape)(X)
    #X = tf.keras.layers.ZeroPadding2D((3,3))(X_input)#This layer can add rows and columns of zeros at the top, bottom, left and right side of an image tensor.

    #X_input = tf.keras.layers.Input(shape=(32, 32, 2))
    #X = tf.keras.layers.ZeroPadding2D(padding=(3, 3))(X)

    Z = tf.layers.conv2d(X, 64, (7, 7), strides=(1, 1), padding='same',
                         kernel_initializer=tf.contrib.layers.xavier_initializer(seed=0), name='conv1')
    Z = tf.layers.batch_normalization(Z, training=is_training, momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON,
                                      name='bn_conv1')
    Z = tf.keras.layers.LeakyReLU(_LEAKY_ALPHA, name="leaky_relu")(Z)
    
    Z = tf.keras.layers.MaxPooling2D((3, 3), strides=(1, 1))(Z)
       
    # Stage2
    Z = convolutional_block(Z, f=3, filters=[64, 64, 256], stage=2, block='a', is_training=is_training, s=1)
    Z = identity_block(Z, 3, [64, 64, 256], stage=2, block='b', is_training=is_training)
    Z = identity_block(Z, 3, [64, 64, 256], stage=2, block='c', is_training=is_training)
    
    # Stage 3 (≈4 lines)
    Z = convolutional_block(Z, f=3, filters=[128, 128, 512], stage=3, block='a', is_training=is_training, s=2)
    Z = identity_block(Z, 3, [128, 128, 512], stage=3, block='b', is_training=is_training)
    Z = identity_block(Z, 3, [128, 128, 512], stage=3, block='c', is_training=is_training)
    Z = identity_block(Z, 3, [128, 128, 512], stage=3, block='d', is_training=is_training)
    
    # Stage 4 (≈6 lines)
    Z = convolutional_block(Z, f=3, filters=[256, 256, 1024], stage=4, block='a', is_training=is_training, s=2)
    Z = identity_block(Z, 3, [256, 256, 1024], stage=4, is_training=is_training, block='b')
    Z = identity_block(Z, 3, [256, 256, 1024], stage=4, is_training=is_training, block='c')
    Z = identity_block(Z, 3, [256, 256, 1024], stage=4, is_training=is_training, block='d')
    
    # Stage 5 (≈3 lines)
    Z = convolutional_block(Z, f=3, filters=[512, 512, 2048], stage=5, block='a', is_training=is_training, s=2)
    Z = identity_block(Z, 3, [512, 512, 2048], stage=5, block='b', is_training=is_training)
    Z = identity_block(Z, 3, [512, 512, 2048], stage=5, block='c', is_training=is_training)

    Z = tf.layers.average_pooling2d(Z, pool_size=(2, 2), strides=(2, 2), padding='same', name="averagepooling")
    Z = tf.layers.flatten(Z, name="flatten")

    y_logits = tf.layers.dense(Z, units=classes, name='logits')
    y = tf.nn.softmax(y_logits, name='ybar')
 
    if logits:
        return y, y_logits
    else:
        return y
