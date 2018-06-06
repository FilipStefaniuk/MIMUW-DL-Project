from base_classes.base_model import BaseModel
import tensorflow as tf
import numpy as np

class MyModel(BaseModel):
    def __init__(self, config):
        super(MyModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None, 96, 96, 3])
        self.y = tf.placeholder(tf.float32, shape=[None, 4])

        params = np.load("../deploy.npy", encoding='bytes')
        params = params.tolist()

        #c1 = tf.layers.conv2d(self.x, 32, kernel_size=(5,5), strides=(1,1), name="conv1")
        conv1_param = params["conv1"]
        conv1_weights = tf.Variable(conv1_param[b"weights"], trainable=True)
        conv1_biases = tf.Variable(conv1_param[b"biases"], trainable=True)
        c1 = tf.nn.conv2d(self.x, conv1_weights, strides=[1,1,1,1], padding='SAME', name="conv1")
        c1 = tf.nn.sigmoid(c1 + conv1_biases)

        mp1 = tf.layers.max_pooling2d(c1, pool_size=(2,2), strides=(2,2), name="pool1")

        #c2 = tf.layers.conv2d(mp1, 32, kernel_size=(5,5), strides=(1,1), name="conv2")
        conv2_param = params["conv2"]
        conv2_weights = tf.Variable(conv2_param[b"weights"], trainable=True)
        conv2_biases = tf.Variable(conv2_param[b"biases"], trainable = True)
        c2 = tf.nn.conv2d(mp1, conv2_weights, strides=[1,1,1,1], padding='SAME', name="conv2")
        c2 = tf.nn.sigmoid(c2 + conv2_biases)
        
        mp2 = tf.layers.max_pooling2d(c2, pool_size=(2,2), strides=(2,2), name="pool2")

        reshaped = tf.reshape(mp2, [-1, mp2.shape[1] * mp2.shape[2] * mp2.shape[3]], name="reshape")
        
        #d1 = tf.layers.dense(reshaped, 100, activation=tf.nn.relu, name="ip3")
        d1_param = params["ip3"]
        d1_weights = d1_param[b"weights"]
        d1_biases = d1_param[b"biases"]
        d1 = tf.layers.dense(reshaped, 100, activation=tf.nn.relu, kernel_initializer=tf.constant_initializer(d1_weights), bias_initializer=tf.constant_initializer(d1_biases), trainable=True, name="ip3")
        
        #d2 = tf.layers.dense(d1, 100, activation=tf.nn.relu, name="ip4")
        d2_param = params["ip4"]
        d2_weights = d2_param[b"weights"]
        d2_biases = d2_param[b"biases"]
        d2 = tf.layers.dense(d1, 100, activation=tf.nn.relu, kernel_initializer=tf.constant_initializer(d2_weights), bias_initializer=tf.constant_initializer(d2_biases), trainable=True, name="ip4")
        
        #d3 = tf.layers.dense(d2, 4, name="ip_last")
        d3_param = params["ip_last"]
        d3_weights = d3_param[b"weights"]
        d3_biases = d3_param[b"biases"]
        d3 = tf.layers.dense(d2, 4, kernel_initializer=tf.constant_initializer(d3_weights), bias_initializer=tf.constant_initializer(d3_biases), trainable=True, name="ip_last")
        
        d4 = tf.layers.dense(d3, 4, kernel_initializer=tf.constant_initializer([[0,1,0,1],[1,0,0,0],[0,0,1,0],[0,0,0,0]]), trainable=True, name="ip_last2")


        self.labels = tf.argmax(self.y, 1)
        self.predictions = tf.argmax(d4, 1)


        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d4))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss,
                                                                                         global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(d4, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            
        
        
        
        #.conv(5, 5, 32, 1, 1, relu=False, name='conv1')
        #     .max_pool(2, 2, 2, 2, name='pool1')
        #     .relu(name='relu1')
        #     .conv(5, 5, 32, 1, 1, relu=False, name='conv2')
        #     .max_pool(2, 2, 2, 2, name='pool2')
        #     .relu(name='relu2')
        #     .fc(100, name='ip3')
        #     .fc(100, name='ip4')
        #     .fc(4, relu=False, name='ip_last')
        #     .softmax(name='prob')) ?????????
        

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        
        
