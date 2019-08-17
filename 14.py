import tensorflow as tf
import numpy as np

#File read
filename_queue=tf.train.string_input_producer(\
                ['test_score.csv'],shuffle=False, name='filename_queue')
reader=tf.TextLineReader()
key, value=reader.read(filename_queue)

#1x4 matrix
record_default=[[0.0],[0.0],[0.0],[0.0]]

xy=tf.decode_csv(value,record_defaults=record_default)

train_x_batch, train_y_batch=tf.train.batch([xy[:-1], xy[-1:]],batch_size=10)
#File read Done

X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

#Layer_1
W_1=tf.Variable(tf.random_normal([3,10]),name="Weight_1")
b_1=tf.Variable(tf.random_normal([10]), name="Bias_1")
layer_1=tf.matmul(W_1,X)+b_1

#Layer_2
W_2=tf.Variable(tf.random_normal([10,1]),name="Weight_2")
b_2=tf.Variable(tf.random_normal([1]), name="Bias_2")
H=tf.matmul(layer_1, W_2)+b_2

cost=tf.reduce_mean(tf.square(H-Y))
train=tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    x_data, y_data = sess.run([train_x_batch, train_y_batch])
    for step in range(2001):
        sess.run(train, feed_dict={X:x_data, Y:y_data})
        
coord.request_stop()
coord.join(threads)
        