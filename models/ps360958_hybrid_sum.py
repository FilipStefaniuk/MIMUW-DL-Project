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

        self.x = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])
        self.y = tf.placeholder(tf.float32, shape=[None, 2])
        
        with tf.name_scope('external'):
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
        with tf.name_scope('ps360958_5_part'):
            with tf.variable_scope("ps360958_5_part") as scope0:
                c1 = tf.layers.conv2d(conv45, 256, (5,5), activation=tf.nn.relu, padding="SAME", name="conv9")
                bn5 = tf.layers.batch_normalization(c1, name="bn5")
                mp1 = tf.layers.max_pooling2d(bn5, 2, 2, name="maxpool4")
                c2 = tf.layers.conv2d(mp1, 128, (5,5), activation=tf.nn.relu, padding="SAME", name="conv10")
                bn6 = tf.layers.batch_normalization(c2, name="bn6")
                mp2 = tf.layers.max_pooling2d(bn6, 2, 2, name="maxpool5")
                c3 = tf.layers.conv2d(mp2, 64, (3,3), activation=tf.nn.relu, padding="SAME", name="conv11")
                mp3 = tf.layers.max_pooling2d(c3, 2, 2, name="maxpool6")
                c4 = tf.layers.conv2d(mp3, 32, (3,3), activation=tf.nn.relu, padding="SAME", name="conv12")
                mp4 = tf.layers.max_pooling2d(c4, 2, 2, name="maxpool7")
                # network architecture
                reshaped = tf.reshape(mp4, [-1, mp4.shape[1] * mp4.shape[2] * mp4.shape[3]], name="reshape")
                d0 = tf.layers.dense(reshaped, 64, activation=tf.nn.relu, name="dense0")
                d1 = tf.layers.dense(reshaped, 32, activation=tf.nn.relu, name="dense1")
                d2 = tf.layers.dense(d1, 2, name="dense2")
                signal_5 = d2
        
        with tf.name_scope('ps360958_6_part'):
            with tf.variable_scope("ps360958_6_part") as scope0:
                c1 = tf.layers.conv2d(conv45, 512, (5,5), activation=tf.nn.relu, padding="SAME", name="conv9")
                bn5 = tf.layers.batch_normalization(c1, name="bn5")
                mp1 = tf.layers.max_pooling2d(bn5, 2, 2, name="maxpool4")
                c2 = tf.layers.conv2d(mp1, 256, (5,5), activation=tf.nn.relu, padding="SAME", name="conv10")
                bn6 = tf.layers.batch_normalization(c2, name="bn6")
                mp2 = tf.layers.max_pooling2d(bn6, 2, 2, name="maxpool5")
                c3 = tf.layers.conv2d(mp2, 128, (3,3), activation=tf.nn.relu, padding="SAME", name="conv11")
                bn7 = tf.layers.batch_normalization(c3, name="bn7")
                mp3 = tf.layers.max_pooling2d(bn7, 2, 2, name="maxpool6")
                c4 = tf.layers.conv2d(mp3, 64, (3,3), activation=tf.nn.relu, padding="SAME", name="conv12")
                mp4 = tf.layers.max_pooling2d(c4, 2, 2, name="maxpool7")
                # network architecture
                reshaped = tf.reshape(mp4, [-1, mp4.shape[1] * mp4.shape[2] * mp4.shape[3]], name="reshape")
                d0 = tf.layers.dense(reshaped, 128, activation=tf.nn.relu, name="dense0")
                bn8 = tf.layers.batch_normalization(d0, name="bn8")
                d1 = tf.layers.dense(bn8, 64, activation=tf.nn.relu, name="dense1")
                d2 = tf.layers.dense(d1, 64, activation=tf.nn.relu, name="dense2")
                d3 = tf.layers.dense(d2, 2, name="dense3")
                signal_6 = d3
                
        with tf.variable_scope("ps360958_8_part")as scope0:
            with tf.name_scope('ps360958_8_part'):
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

                signal_8 = d2

        signal = signal_5 + signal_6 + signal_8
        self.labels = self.y
        self.predictions = signal
        

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=signal))
            #self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss,
            #                                                                             global_step=self.global_step_tensor,
                                                                                        #var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ps360958_5_part'))
            correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

