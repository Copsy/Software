import tensorflow as tf

filename_queue=tf.train.string_input_producer(['diabets.csv'],shuffle=False,name="filename_queue")
reader=tf.TextLineReader()
key, value=reader.read(filename_queue)

reader_default_format=[[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]]
xy=tf.decode_csv(value,record_defaults=reader_default_format)

train_x_batch, train_y_batch=tf.train.batch([xy[0:-1], xy[-1:]],batch_size=10)

X=tf.placeholder(tf.float32, shape=[None, 8])
Y=tf.placeholder(tf.float32, shape=[None, 1])

W=tf.Variable(tf.random_normal([8,1]),name="Weight")
b=tf.Variable(tf.random_normal([1]), name="Bias")

H=tf.sigmoid(tf.matmul(X,W)+b)
cost=tf.reduce_mean(Y*tf.log(H)+(1-Y)*tf.log(1-H))*-1
train=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predict=tf.cast(H>0.5, dtype=tf.float32)
accuracy=tf.reduce_mean(tf.cast(tf.equal(predict, Y),dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    coord=tf.train.Coordinator()
    threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    
    for step in range (20001):
        x_batch,y_batch=sess.run([train_x_batch, train_y_batch])
        a,_=sess.run([accuracy,train], feed_dict={X:x_batch, Y:y_batch})
        if step % 100 ==0:
            print(step,sess.run(cost, feed_dict={X:x_batch, Y:y_batch}))
    coord.request_stop()
    coord.join(threads)