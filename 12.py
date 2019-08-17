from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pylab as plt
import random

mnist=input_data.read_data_sets("MNIST_DATA/",one_hot=True)

r=random.randint(0,mnist.test.num_examples-1)
nb_class=512
final_nb_class=10
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)
#Deep Wide Training
W1=tf.Variable(tf.random_normal([784,nb_class]),name="Weight_1")
b1=tf.Variable(tf.random_normal([nb_class]),name="Bias_1")
logistic_1=tf.matmul(X,W1)+b1
layer_1=tf.nn.relu(logistic_1)

W2=tf.Variable(tf.random_normal([nb_class,nb_class]),name="Weight_2")
b2=tf.Variable(tf.random_normal([nb_class]),name="Bias_2")
logistic_2=tf.matmul(layer_1,W2)+b2
layer_2=tf.nn.relu(logistic_2)

W3=tf.Variable(tf.random_normal([nb_class,nb_class]),name="Weight_3")
b3=tf.Variable(tf.random_normal([nb_class]),name="Bias_3")
logistic_3=tf.matmul(layer_2,W3)+b3
layer_3=tf.nn.relu(logistic_3)

W4=tf.Variable(tf.random_normal([nb_class,nb_class]),name="Weight_4")
b4=tf.Variable(tf.random_normal([nb_class]),name="Bias_4")
logistic_4=tf.matmul(layer_3,W4)+b4
layer_4=tf.nn.relu(logistic_4)

W5=tf.Variable(tf.random_normal([nb_class,final_nb_class]),name="Weight_5")
b5=tf.Variable(tf.random_normal([final_nb_class]),name="Bias_5")
logistic_5=tf.matmul(layer_4,W5)+b5
H=tf.nn.softmax(logistic_5)
'''
nb_class=10
#mnist.test.num_examples==10000
r=random.randint(0,mnist.test.num_examples-1)
#28x28==784
X=tf.placeholder(tf.float32, shape=[None, 784])
#Y is one_hot
Y=tf.placeholder(tf.float32, shape=[None,nb_class])

W=tf.Variable(tf.random_normal([784,nb_class]),name="Weight")
b=tf.Variable(tf.random_normal([nb_class]),name="Bias")
#Soft max--->
logistic=tf.matmul(X,W)+b
H=tf.nn.softmax(logistic)
'''
cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(H),axis=1))
train=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predict=tf.arg_max(H,1)
is_correct=tf.equal(predict, tf.arg_max(Y,1))
accuracy=tf.reduce_mean(tf.cast(is_correct,tf.float32))
#Soft Max is done

epoch=15
batch_size=100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for term in range(epoch):
        total_batch=int(mnist.train.num_examples/batch_size)
        
        for step in range(total_batch):
            batch_xs, batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train,feed_dict={X:batch_xs, Y:batch_ys})
            
        print("Epooch : %04d"%(term+1))
    print("Label : ", sess.run(tf.arg_max(mnist.test.labels[r:r+1],1)))
    
    plt.imshow(mnist.test.images[r:r+1].reshape(28,28),cmap="Greys",interpolation="nearest")
    plt.show()