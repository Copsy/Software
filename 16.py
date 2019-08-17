from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pylab as plt
import random
import numpy as np

# -->shape [data#, row, cols, depth]
img=np.array([[[[1],[2],[3]],\
              [[4],[5],[6]],\
              [[7],[8],[9]]]],dtype=np.float32)

#2x2x1x1
#--->shape [row, cols, depth, filter#]
filter_1=np.array([[[[1,1,1]],[[1,1,1]]],\
                   [[[1,1,1]],[[1,1,1]]]])

pool=tf.nn.max_pool(img,ksize=[1,2,2,1],strides=[1,1,1,1],padding="VALID")

with tf.Session() as sess:
    conv2d=tf.nn.conv2d(img,filter_1,strides=[1,1,1,1],padding="VALID")
    conv2d_img=conv2d.eval()
    conv2d_img=np.swapaxes(conv2d_img,0,3)
    print(pool.eval())
    for i, one_img in enumerate(conv2d_img):
        plt.subplot(1,3,i+1); plt.imshow(one_img.reshape(2,2), cmap="gray")