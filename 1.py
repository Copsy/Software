import tensorflow as tf

x_data=[1,2,3]
y_data=[1,2,3]

W=tf.Variable(tf.random_normal([1]), name='weight')
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

H=W*X

cost=tf.reduce_mean(tf.square(H-Y))
'''

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.1)
train=optimizer.minimize(cost)

'''

learning_rate=0.1
gradient=tf.reduce_mean((H-Y)*X)
descent=W-learning_rate*gradient
update=W.assign(descent)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(21):
    sess.run(update,feed_dict={X:x_data, Y:y_data})
    print(i,sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W))