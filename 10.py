import tensorflow as tf
import numpy as np

xy=np.loadtxt("zoo.csv", delimiter=',',dtype=np.float32)

x_data=xy[:,0:-1]
y_data=xy[:,[-1]]

#Class are 6 

# X is N x 16 matrix
X=tf.placeholder(tf.float32, shape=[None, 16])
# Y is N x 1 matrix
Y=tf.placeholder(tf.int32, shape=[None, 1])
# Y_one_hot is N x 1 x 7 matrix
Y_one_hot=tf.one_hot(Y, 7)
# Y_one_hot is N x 7 matrix
# Y_one_hot is real answer
Y_one_hot=tf.reshape(Y_one_hot,[-1, 7])

W=tf.Variable(tf.random_normal([16,7]),name="Weight")
b=tf.Variable(tf.random_normal([7]), name="Bias")

#H=XW+b
logistic=tf.matmul(X,W)+b
#possibility axis 1 is Row
#exp(logistic)/Sum(exp(logistic))
#H is N x 7 matrix
H=tf.nn.softmax(logistic, axis=1)

cost_i=tf.nn.softmax_cross_entropy_with_logits(logits=logistic,labels=Y_one_hot)
cost=tf.reduce_mean(cost_i)
train=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

#Flag 1 is Axis 1 mean col
prediction=tf.arg_max(H, 1)
correct_prediction=tf.equal(prediction, tf.arg_max(Y_one_hot,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess=tf.Session()
sess.run(tf.global_variables_initializer())
for step in range (2001) :
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 10 == 0:
        loss, acc=sess.run([cost, accuracy],feed_dict={X:x_data, Y:y_data})
        print("STEP {:5}\tLoss {:.3}\tAcc {:.3%}".format(step, loss, acc))