from base_classes.base_model import BaseModel
import tensorflow as tf


class ExampleModel(BaseModel):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None, 720, 1280, 3])
        self.y = tf.placeholder(tf.float32, shape=[None, 4])
        
        reshaped = tf.reshape(self.x, [-1, 2764800], name="reshape")

        # network architecture
        d1 = tf.layers.dense(reshaped, 64, activation=tf.nn.relu, name="dense1")
        d2 = tf.layers.dense(d1, 4, name="dense2")

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d2))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss,
                                                                                         global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

