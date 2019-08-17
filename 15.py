from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pylab as plt
import random


mnist=input_data.read_data_sets("MNIST_DATA/",one_hot=True)
#mnist.text.num_examples==10000
r=random.randint(0,mnist.test.num_examples-1)
nb_class=512
final_nb_class=10
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)
keep_prob=tf.placeholder(tf.float32)

#Layer_1
#W_1=tf.Variable(tf.random_normal(shape=[784,nb_class]),name="Weight_1")
#Using Xavier_Initializer()
W_1=tf.get_variable("Weight_1",shape=[784, nb_class],initializer=tf.contrib.layers.xavier_initializer())
b_1=tf.Variable(tf.random_normal(shape=[nb_class]), name="Bias_1")
logistic_1=tf.matmul(X,W_1)+b_1
#Using ReLU
_layer_1=tf.nn.relu(logistic_1)
layer_1=tf.nn.dropout(_layer_1,keep_prob=keep_prob)

#Layer_2
#W_2=tf.Variable(tf.random_normal(shape=[nb_class, nb_class]), name="Weight_2")
    
W_2=tf.get_variable("Weight_2", [nb_class, nb_class],initializer=tf.contrib.layers.xavier_initializer())
b_2=tf.Variable(tf.random_normal(shape=[nb_class]), name="Bias_2")
logistic_2=tf.matmul(layer_1, W_2)+b_2
_layer_2=tf.nn.relu(logistic_2)
layer_2=tf.nn.dropout(_layer_2,keep_prob=keep_prob)
    
#W_3=tf.Variable(tf.random_normal(shape=[nb_class, final_nb_class]), name="Weight_3")
    
W_3=tf.get_variable("Weight_3", [nb_class, final_nb_class], initializer=tf.contrib.layers.xavier_initializer())
b_3=tf.Variable(tf.random_normal(shape=[final_nb_class]), name="Bias_3")
logistic_3=tf.matmul(layer_2, W_3)+b_3
H=tf.nn.softmax(logistic_3)
    
cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(H),axis=1))
train=tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)
    
predict=tf.arg_max(H, 1)
    
epoch=20
batch_size=100
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(mnist.test.num_examples)
    for step in range(epoch):
        total_batch=int(mnist.train.num_examples/batch_size)
        for term in range(total_batch):
            batch_x, batch_y=mnist.train.next_batch(batch_size)
            sess.run(train, feed_dict={X:batch_x, Y:batch_y, keep_prob:0.5})
            
        c=sess.run(cost,feed_dict={X:batch_x, Y:batch_y, keep_prob:1.0})
        
        print("Epooch : ",step+1,"--> Cost : ",c)
            
    print("Label : ", sess.run(tf.arg_max(mnist.test.labels[r:r+1],1)))
        
    plt.imshow(mnist.test.images[r:r+1].reshape(28,28),cmap="Greys",interpolation="nearest")
    plt.show()