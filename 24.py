import tensorflow as tf
import cv2 as cv
import numpy as np
import matplotlib.pylab as plt
import keras
import glob
'''
USING CNN
Input_Layer->Hidden_Layer_1->Hidden_Layer_2\
             Calculate_of_loss->Output_Layer(Fully_Connected)->Result

Hidden_Layer_1 : Convolution_1->SubSampling_1
Hidden_Layer_2 : Convolution_2->SubSampling_2
'''

x=tf.placeholder(tf.float32,shape=[None, 120*320])
#Image size --> 320x120
y=tf.placeholder(tf.float32,shape=[None, 1])
#Label --> List : Palm
X_img=tf.reshape(x, shape=[-1,120,320,1])
X=[]
Y=[]

#DATA load
train_data=glob.glob("d:/01_palm/*.png")

for i, data in enumerate(train_data):
    img=cv.imread(data)
    img=cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img=cv.resize(img, (120,320))
    X.append(img)
    Y.append("Palm")
#2000x320x120--->Data # is 2000, 320x120
X=np.array(X, dtype=np.uint8)
Y=np.array(Y)

#Dropout rate : 0.5
drop=0.5

#Linear Regression
'''
#Layer_1 : Convolution_1 -> Subsampling
conv_1=tf.layers.conv2d(inputs=X_img,filters=32,kernel_size=[3,3],padding="SAME", activation=tf.nn.relu)
pool_1=tf.layers.max_pooling2d(inputs=conv_1,pool_size=[2,2],padding="SAME",strides=2)
#Pool_1 : -1 60 160 32

dropout_1=tf.layers.dropout(inputs=pool_1,rate=drop,training=True)

conv_2=tf.layers.conv2d(inputs=dropout_1,filters=64, kernel_size=[3,3],padding="SAME",activation=tf.nn.relu)
pool_2=tf.layers.max_pooling2d(inputs=conv_2,pool_size=[2,2],padding="SAME", strides=2)

dropout_2=tf.layers.dropout(inputs=pool_2, rate=drop, training=True)
# ? 30, 80, 64 tf.float32
#Dense
flat=tf.reshape(dropout_2, [-1, 30*80*64])
dense_1=tf.layers.dense(inputs=flat,units=625,activation=tf.nn.relu)
dropout_3=tf.layers.dropout(inputs=flat,rate=drop, training=True)
logistic=tf.layers.dense(inputs=dropout_3,units=1)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logistic,labels=Y))
optimizer=tf.train.AdadeltaOptimizer(learning_rate=0.01).minimize(cost)
'''