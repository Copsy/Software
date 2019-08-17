from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pylab as plt
import random
import numpy as np

mnist=input_data.read_data_sets("MNIST_DATA/",one_hot=True)
#mnist.text.num_examples==10000

#Train Image size is 28x28
img=mnist.train.images[0].reshape(28,28)
plt.imshow(img,cmap="gray")

sess=tf.InteractiveSession()
# DATA number, row,cols, Depth 
img=img.reshape(-1, 28,28,1)

# Filter--> Rows,Cols, Depth,Filter# 
filter_1=tf.Variable(tf.random_normal([3,3,1,5],stddev=0.01))
# size==28/2
conv=tf.nn.conv2d(img,filter_1,strides=[1,2,2,1],padding="SAME")
sess.run(tf.global_variables_initializer())
conv_img=conv.eval()
conv_img=np.swapaxes(conv_img,0,3)

#The number of filter is 5
for i, one_img in enumerate(conv_img):
    plt.subplot(1,5,i+1); plt.imshow(one_img.reshape(14,14),cmap="gray")

#MAX pooling-->conv is 14x14x5--->strides is 2 so pool size is 7x7
pool=tf.nn.max_pool(conv, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
#=sess.run(tf.global_variables_initializer())
pool_img=pool.eval()
pool_img=np.swapaxes(pool_img, 0, 3)
for i, one_img in enumerate(pool_img):
    plt.subplot(1,5,i+1); plt.imshow(one_img.reshape(7,7), cmap="gray")