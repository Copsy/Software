import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learing_rate=0.001
epoch=10
batch_size=100
dropout_rate=0.5

#Define Class for easy making

class Model:
    #Member function
    
    def __init__(self, sess, name):
        self.sess=sess
        self.name=name
        self._build_net()#-->Making Net
        
        '''
            NET : DATA->convolution->Subsampling->Convolution->Subsampling->fully_connected
        '''
    def _build_net(self):
        with tf.variable_scope(self.name):
            #Decision for wherther it is test or training
            self.training=tf.placeholder(tf.bool)
            self.X=tf.placeholder(tf.float32,shape=[None, 784])
            X_img=tf.reshape(self.X,[-1,28,28,1])#--> # : None, 28x28x1 matrix(gray image)
            self.Y=tf.placeholder(tf.float32,shape=[None, 10])            
            
            #Layer_1
            #Convolution_1
            conv_1=tf.layers.conv2d(inputs=X_img,filters=32,kernel_size=[3,3],\
                                    padding="SAME",activation=tf.nn.relu)
            '''
                using layers to make easy
                X_img-->28x28x1, # : ?
                filter_1--->[3,3,1,32] : 3x3x1, # : 32
                conv_1--->28x28x32
            '''
            #Padding_1
            pad_1=tf.layers.max_pooling2d(inputs=conv_1, pool_size=[2,2],\
                                          padding="SAME", strides=2)
            '''
                pool_size -->[2,2], strides [2,2]
                pad_1--->14x14x32
            '''
            drop_1=tf.layers.dropout(inputs=pad_1, rate=dropout_rate,training=self.training)
            
            conv_2=tf.layers.conv2d(inputs=drop_1,filters=64,kernel_size=[3,3],\
                                    padding="SAME", activation=tf.nn.relu)
            '''
                drop_1--->14x14x32
                filter_2--->[3,3,32,64] : 3x3x32, # : 64
                conv_2--->14x14x64
            '''
            pad_2=tf.layers.max_pooling2d(inputs=conv_2,pool_size=[2,2],padding="SAME",strides=2)
            '''
                pool_size=[2,2], strides=[2,2]
                pad_2=7x7x64
            '''
            drop_2=tf.layers.dropout(inputs=pad_2,rate=dropout_rate, training=self.training)
            
            #Dense Layers
            
            flat=tf.reshape(drop_2,shape=[-1,7*7*64])
            #Output # : 625
            dense=tf.layers.dense(inputs=flat,units=625,activation=tf.nn.relu)
            drop_3=tf.layers.dropout(inputs=dense,rate=dropout_rate, training=self.training)
            #Output # : 10
            self.logistic=tf.layers.dense(inputs=drop_3,unit=10)
            
        self.cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logistic,labels=self.Y))
        self.optimizer=tf.train.AdamOptimizer(learing_rate=self.learing_rate).minimize(self.cost)
        
        correct_prediction = tf.equal(
            tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
    def predict(self,x_test,training=False):
        return self.sess.run(self.logistic,feed_dict={self.X : x_test, self.training : training})
        
    def train(self,x_data,y_data,training=True):
        return self.sess.run([self.cost,self.optimizer],feed_dict={self.X : x_data,self.Y : y_data, self.training : training})
    
    
with tf.Session() as sess:
    m1=Model(sess, "M1")
    sess.run(tf.global_variables_initializer())
    print("Learing start")
    for step in range(epoch):
        avg_cost=0
        total_batch=int(mnist.train.num_examples/batch_size)
        
        for i in range(total_batch):
            batch_x, batch_y=mnist.train.next_batch(batch_size)
            c,_=m1.train(batch_x,batch_y)
            avg_cost+=c/total_batch
            
        print("Epoch : ",step," AVG_COST : ",avg_cost)