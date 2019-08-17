import tensorflow as tf

train_x_data=[[1,2,1],[1,3,2],[1,3,4],[1,5,5],[1,7,5],[1,2,5],[1,6,6],[1,7,7]]
train_y_data=[[0,0,1],[0,0,1],[0,0,1],[0,1,0],[0,1,0],[0,1,0],[1,0,0],[1,0,0]]

test_x=[[2,1,1],[3,1,2],[3,3,4],[1,7,7]]
test_y=[[0,0,1],[0,0,1],[0,0,1],[1,0,0]]

X=tf.placeholder(tf.float32, shape=[None,3])
Y=tf.placeholder(tf.float32, shape=[None,3])

W=tf.Variable(tf.random_normal([3,3]),name="Weight")
b=tf.Variable(tf.random_normal([3]), name="Bias")

logistic=tf.matmul(X,W)+b
#Make possibility ( 0 to 1 ) All sum is 1
H=tf.nn.softmax(logistic,axis=1)
cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(H),axis=1))
train=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predict=tf.arg_max(H,1)
is_correct=tf.equal(predict,tf.arg_max(Y ,1))
accuracy=tf.reduce_mean(tf.cast(is_correct,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        sess.run(train, feed_dict={X:train_x_data, Y:train_y_data})
        
        if step % 20 == 0:
            co, ac, h = sess.run([cost, accuracy, H], feed_dict={X:train_x_data, Y:train_y_data})
            print("Cost : ",co,"\tAccuracy : ",ac,"\nH : \n",h)
            
    label=sess.run(predict,feed_dict={X:test_x})
    print("\n",label)
    