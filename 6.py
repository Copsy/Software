import tensorflow as tf

x_data=[[100,100,100],[50,60,70],[70,80,90]]
y_data=[[100],[67],[89]]

X=tf.placeholder(tf.float32,shape=[None,3])
Y=tf.placeholder(tf.float32,shape=[None,1])

W=tf.Variable(tf.random_normal([3,1]), name="Weight")
b=tf.Variable(tf.random_normal([1]), name="bias")

H=tf.matmul(X,W)+b

cost=tf.reduce_mean(tf.square(H-Y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.glorot_normal_initializer())

for step in range(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})

