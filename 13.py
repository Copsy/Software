import numpy as np
import tensorflow as tf

train_x_data=np.array([[0,0],[1,0],[0,1],[1,1]],dtype=np.float32)
train_y_data=np.array([[0],[1],[1],[0]],dtype=np.float32)

test_x=np.array([[1,1],[0,0],[1,0],[0,1]],dtype=np.float32)
test_y=np.array([[0],[0],[1],[1]], dtype=np.float32) 
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)
'''
W=tf.Variable(tf.random_normal([2,1]),name="Weight")
b=tf.Variable(tf.random_normal([1]), name="Bias")
#1/1+exp(-z)
H=tf.sigmoid(tf.matmul(X,W)+b)
'''

W1=tf.Variable(tf.random_normal([2,10]),name="Weight_1")
b1=tf.Variable(tf.random_normal([10]),name="Bias_1")
logistic_1=tf.matmul(X,W1)+b1
layer_1=tf.sigmoid(logistic_1)

W2=tf.Variable(tf.random_normal([10,10]),name="Weight_2")
b2=tf.Variable(tf.random_normal([10]),name="Bias_2")
logistic_2=tf.matmul(layer_1,W2)+b2
layer_2=tf.sigmoid(logistic_2)

W3=tf.Variable(tf.random_normal([10,10]),name="Weight_3")
b3=tf.Variable(tf.random_normal([10]),name="Bias_3")
logistic_3=tf.matmul(layer_2,W2)+b2
layer_3=tf.sigmoid(logistic_3)

W2=tf.Variable(tf.random_normal([10,10]),name="Weight_4")
b2=tf.Variable(tf.random_normal([10]),name="Bias_4")
logistic_4=tf.matmul(layer_1,W2)+b2
layer_4=tf.sigmoid(logistic_4)

W2=tf.Variable(tf.random_normal([10,10]),name="Weight_5")
b2=tf.Variable(tf.random_normal([10]),name="Bias_5")
logistic_5=tf.matmul(layer_1,W2)+b2
layer_5=tf.sigmoid(logistic_5)

W2=tf.Variable(tf.random_normal([10,1]),name="Weight_6")
b2=tf.Variable(tf.random_normal([1]),name="Bias_6")
logistic_6=tf.matmul(layer_1,W2)+b2
H=tf.sigmoid(logistic_6)

cost=-tf.reduce_mean(Y*tf.log(H)+(1-Y)*tf.log(1-H))
train=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

predict=tf.cast(H>0.5, dtype=tf.float32)
accuracy=tf.reduce_mean(tf.cast(tf.equal(predict,Y),dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for step in range(10001):
        sess.run(train, feed_dict={X:train_x_data, Y:train_y_data})

    h,c,a=sess.run([H,predict,accuracy],feed_dict={X:train_x_data, Y:train_y_data})
    print(h,"\n",c,"\n",a)