import tensorflow as tf

x_data=[[1,2,1,1],[2,1,3,2],[3,1,3,4],[4,1,5,5],[1,7,5,5],[1,2,5,6],[1,6,6,6],[1,7,7,7]]
y_data=[[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]

X=tf.placeholder("float", shape=[None,4])
Y=tf.placeholder("float", shape=[None,3])

W=tf.Variable(tf.random_normal([4,3]),name="Weight")
b=tf.Variable(tf.random_normal([3]),name="Bias")

logistic=tf.matmul(X,W)+b
H=tf.nn.softmax(logistic,axis=1)

cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(H),axis=1))
train=tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        sess.run(train,feed_dict={X:x_data, Y:y_data})
        
        if step % 10 == 0:
            print(step,"\n",sess.run(cost,feed_dict={X:x_data,Y:y_data}))
            
    a=sess.run(H,feed_dict={X:[[1,11,7,9]]})
    print("\n",a,sess.run(tf.arg_max(a,1)))