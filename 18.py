from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist=input_data.read_data_sets("MNIST_DATA/",one_hot=True)
epoch=10
batch_size=100
'''
DATA->Convolution->Subsampling->Convolution->Subsampling->fully-connected
'''
#Data input
X=tf.placeholder(tf.float32, shape=[None, 784])
#change to image format -->-1 : None, 28x28x1-->Gray Color
X_img=tf.reshape(X, [-1,28,28,1])
Y=tf.placeholder(tf.float32,shape=[None, 10])

#Layer_1
#Convolution Layer
#Filter size 3x3x1 --> # is 32
filter_1=tf.Variable(tf.random_normal([3,3,1,32],stddev=0.01))
#Logistic_1--> 28x28x32-->the number is 32
logistic_1=tf.nn.conv2d(X_img, filter_1,strides=[1,1,1,1],padding="SAME")
layer_1=tf.nn.relu(logistic_1)

#Subsampling_1
#layer_1-->size is 14x14x1--->32
layer_1=tf.nn.max_pool(layer_1, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

#Layer_2
#Convolution_2, previous output size is 14x14x1, the number of that is 32
filter_2=tf.Variable(tf.random_normal([3,3,32,64],stddev=0.01))
#logistic_2 size -->14x14x64--># : 64
logistic_2=tf.nn.conv2d(layer_1, filter_2, strides=[1,1,1,1],padding="SAME")
layer_2=tf.nn.relu(logistic_2)

#Subsampling_2
#Layer_2---> 14x14x64
layer_2=tf.nn.max_pool(layer_2, ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")
#layer_2---> 7x7x64
layer_2=tf.reshape(layer_2,[-1,7*7*64])
#layer_2---> nx3136

#Fully connected
with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE) as scope:
    W=tf.get_variable("Weight_1",shape=[7*7*64,10], initializer=tf.contrib.layers.xavier_initializer())
    b=tf.Variable(tf.random_normal([10]))
    H=tf.matmul(layer_2, W)+b

    cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=H, labels=Y))
    train=tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for term in range(epoch):
            avg_cost=0
            total_batch=int(mnist.train.num_examples/batch_size)
            for i in range(total_batch):
                batch_x, batch_y=mnist.train.next_batch(batch_size)
                feed_dict={X:batch_x, Y:batch_y}
                c,_=sess.run([cost, train], feed_dict=feed_dict)
                avg_cost+=c/total_batch
            print("Epoch : %04d" %(term+1)," Cost : ","{:.9f}".format(avg_cost))
        