from base_classes.base_model import BaseModel
import tensorflow as tf


class MyModel(BaseModel):
    def __init__(self, config):
        super(MyModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None, 96, 96, 3])
        self.y = tf.placeholder(tf.float32, shape=[None, 4])
        
        c1 = tf.layers.conv2d(self.x, 32, kernel_size=(5,5), strides=(1,1), name="conv1")
        mp1 = tf.layers.max_pooling2d(c1, pool_size=(2,2), strides=(2,2), name="pool1")
        c2 = tf.layers.conv2d(mp1, 32, kernel_size=(5,5), strides=(1,1), name="conv2")
        mp2 = tf.layers.max_pooling2d(c2, pool_size=(2,2), strides=(2,2), name="pool2")
        
        reshaped = tf.reshape(mp2, [-1, mp2.shape[1] * mp2.shape[2] * mp2.shape[3]], name="reshape")
        d1 = tf.layers.dense(reshaped, 100, activation=tf.nn.relu, name="ip3")
        d2 = tf.layers.dense(d1, 100, activation=tf.nn.relu, name="ip4")
        d3 = tf.layers.dense(d2, 4, name="ip_last")
        
        self.labels = tf.argmax(self.y, 1)
        self.predictions = tf.argmax(d3, 1)
        
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d3))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss,
                                                                                         global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(d3, 1), tf.argmax(self.y, 1))
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
        
        
