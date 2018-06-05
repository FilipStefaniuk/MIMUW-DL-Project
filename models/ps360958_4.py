from base_classes.base_model import BaseModel
import tensorflow as tf


class MyModel(BaseModel):
    def __init__(self, config):
        super(MyModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
        self.y = tf.placeholder(tf.float32, shape=[None, 4])
        
        with tf.variable_scope("external")as scope0:
            with tf.name_scope("down_1") as scope:
                conv1 = tf.layers.conv2d(self.x, 64, (3,3), padding="SAME", activation = tf.nn.relu, name = "conv1")
                bn1 = tf.layers.batch_normalization(conv1, name="bn1")
                pool1 = tf.layers.max_pooling2d(bn1, 2, 2, name = "pool1")
            with tf.name_scope("down_2") as scope:
                conv2 = tf.layers.conv2d(pool1, 64, (3,3), padding="SAME", activation = tf.nn.relu,name = "conv2")
                bn2 = tf.layers.batch_normalization(conv2, name="bn2")
                pool2 = tf.layers.max_pooling2d(bn2, 2, 2, name = "pool2")
            with tf.name_scope("down_3") as scope:
                conv3 = tf.layers.conv2d(pool2, 128, (3,3), padding="SAME", activation = tf.nn.relu, name = "conv3")
                bn3 = tf.layers.batch_normalization(conv3, name="bn3")
                pool3 = tf.layers.max_pooling2d(bn3, 2, 2, name = "pool3")
            with tf.name_scope("bottom") as scope:
                conv4 = tf.layers.conv2d(pool3, 256, (3,3), padding="SAME", activation = tf.nn.relu, name = "conv4")
                bn4 = tf.layers.batch_normalization(conv4, name="bn4")
                conv45 = tf.layers.conv2d(bn4, 512, (5,5), padding="SAME", activation = tf.nn.relu, name = "conv45")

        with tf.variable_scope("new") as scope0:
            c1 = tf.layers.conv2d(conv45, 128, (5,5), activation=tf.nn.relu, padding="SAME", name="conv9")
            bn5 = tf.layers.batch_normalization(c1, name="bn5")
            mp1 = tf.layers.max_pooling2d(bn5, 2, 2, name="maxpool4")
            c2 = tf.layers.conv2d(mp1, 64, (5,5), activation=tf.nn.relu, padding="SAME", name="conv10")
            mp2 = tf.layers.max_pooling2d(c2, 2, 2, name="maxpool5")
            # network architecture
            reshaped = tf.reshape(mp2, [-1, mp2.shape[1] * mp2.shape[2] * mp2.shape[3]], name="reshape")
            d0 = tf.layers.dense(reshaped, 128, activation=tf.nn.relu, name="dense0")
            d1 = tf.layers.dense(reshaped, 64, activation=tf.nn.relu, name="dense1")
            d2 = tf.layers.dense(d1, 4, name="dense2")

        self.labels = tf.argmax(self.y, 1)
        self.predictions = tf.argmax(d2, 1)
        

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d2))
            self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss,
                                                                                         global_step=self.global_step_tensor,
                                                                                        var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='new'))
            correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

