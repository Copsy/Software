import tensorflow as tf

# 3 x 3 matrix
x_data=[[1,2,3],[4,5,6],[7,8,9]]
# 3 x 1 matrix
y_data=[[7],[16],[25]]

X=tf.placeholder(tf.float32, shape=[None, None])
Y=tf.placeholder(tf.float32, shape=[None ,1])

W=tf.Variable(tf.random_normal(shape=[3, 1]), name="Weight")
b=tf.Variable(tf.random_normal(shape=[1]), name="Bias")


#H=w1*x1+w2*x2+w3*x3+b

H=tf.matmul(X,W)+b
cost=tf.reduce_mean(tf.square(H-Y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, H_val, _ = \
    sess.run([cost, H, train], feed_dict={X:x_data, Y:y_data})
    if step % 10 ==0:
        print(step, "\tCost : ",cost_val,"\nPrediction\n",H_val)