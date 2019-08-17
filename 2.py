import tensorflow as tf

x_data=[1,2,3,4]
y_data=[2,5,10,17]

X=tf.placeholder(tf.float32, shape=[None])
Y=tf.placeholder(tf.float32, shape=[None])

a=tf.Variable(tf.random_normal([1]), name='A')
b=tf.Variable(tf.random_normal([1]), name='B')
c=tf.Variable(tf.random_normal([1]), name='C')

H=a*X*X+b*X+c

cost=tf.reduce_mean(tf.square(H-Y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(20001):
    a_val,b_val,c_val, _ =sess.run([a,b,c,train], feed_dict={X:x_data, Y:y_data})
    if step%20==0:
        print(a_val, b_val, c_val)