from base_classes.base_model import BaseModel
import tensorflow as tf


class MyModel(BaseModel):
    def __init__(self, config):
        super(MyModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):

        labels_len = 4 - len(self.config.remove_class)

        self.is_training = tf.placeholder(tf.bool)

        with tf.name_scope('network'):
            self.x = tf.placeholder(tf.float32, shape=[None, self.config.input_size, self.config.input_size, 3])
            self.y = tf.placeholder(tf.float32, shape=[None, labels_len])


            c1 = tf.layers.conv2d(self.x, 8, (5,5), activation=tf.nn.relu, name="conv1")
            bn1 = tf.layers.batch_normalization(c1, name="bn1")
            mp1 = tf.layers.max_pooling2d(bn1, 5, 5, name="maxpool1")
            c2 = tf.layers.conv2d(mp1, 16, (5,5), activation=tf.nn.relu, name="conv2")
            bn2 = tf.layers.batch_normalization(c2, name="bn2")
            mp2 = tf.layers.max_pooling2d(bn2, 5, 5, name="maxpool2")
            # network architecture
            reshaped = tf.reshape(mp2, [-1, mp2.shape[1] * mp2.shape[2] * mp2.shape[3]], name="reshape")
            d1 = tf.layers.dense(reshaped, 64, activation=tf.nn.relu, name="dense1")
            bn3 = tf.layers.batch_normalization(d1, name="bn3")
            d15 = tf.layers.dense(bn3, 32, activation=tf.nn.relu, name="dense1.5")
            d2 = tf.layers.dense(d15, labels_len, name="dense2")

            self.labels = self.y
            self.predictions = d2
        
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d2))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss,
                                                                                         global_step=self.global_step_tensor)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

