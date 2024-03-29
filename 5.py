import tensorflow as tf

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

X=tf.placeholder(tf.float32,shape=[None, 3])
Y=tf.placeholder(tf.float32, shape=[None, 1])

W=tf.Variable(tf.random_normal([3, 1],), name="Weight")
b=tf.Variable(tf.random_normal([1]), name="Bias")

H=tf.matmul(X,W)+b

cost=tf.reduce_mean(tf.square(H-Y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

coord=tf.train.Coordinator()
threads=tf.train.start_queue_runners(sess=sess,coord=coord)

for step in range(2001):
    x_batch,y_batch=sess.run([train_x_batch, train_y_batch])
    cost_val, H_val,_=sess.run([cost, H, train], feed_dict={X:x_batch, Y:y_batch})
    
    if step % 10 ==0:
        print(step, "\nPrediction\n", H_val)
        
coord.request_stop()
coord.join(threads)