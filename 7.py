import tensorflow as tf

x_data=[[1,2,3],[4,5,6],[7,8,9]]
y_data=[[2],[5],[8]]

X=tf.placeholder(tf.float32, shape=[None,3])
Y=tf.placeholder(tf.float32, shape=[None,1])

W=tf.Variable(tf.random_normal(shape=[3,1]), name="Weight")
b=tf.Variable(tf.random_normal(shape=[1]), name="Bias")

#HyperThesis
H=tf.matmul(X,W)+b

cost=tf.reduce_mean(tf.square(H-Y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(50001):
    H_val,W_val,B_val,_=sess.run([H,W,b,train], feed_dict={X:x_data, Y:y_data})
    if step % 20 ==0:
        print(step,"\tW is : ",W_val,"\nBias is ",B_val)