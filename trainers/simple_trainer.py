from base_classes.base_train import BaseTrain
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, roc_auc_score
from texttable import Texttable
from tqdm import tqdm
import tensorflow as tf
import numpy as np


class SimpleTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(SimpleTrainer, self).__init__(sess, model, data, config, logger)
        # self.build_metrics()

    def train_epoch(self):
        
        # self.sess.run(self.reset_metrics)
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        epoch_str = 'Epoch {0}/{1}'.format(cur_epoch, self.config.num_epochs)
        loop = tqdm(range(self.config.num_iter_per_epoch), ncols=120, desc=epoch_str)
        
        losses = []
        accs = []

        
        for _ in loop:
            label, prediction, loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)

        loss = np.mean(losses)
        acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        
        print("============================================")
        print("TRAINING")
        print("============================================")
        print("GLOBAL_STEP {}".format(cur_it))
        print("LOSS: {}".format(loss))
        print("ACCURACY: {}".format(acc))

        summaries_dict = {
            'loss': loss,
            'acc': acc,
        }

        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        
        if self.config.validate:
            self.validate(cur_it)
        
        if self.config.save_model:
            self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = self.sess.run(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, labels, preds, loss, acc = self.sess.run([self.model.train_step, self.model.labels, self.model.predictions, self.model.loss, self.model.accuracy],
                                     feed_dict=feed_dict)
        return labels, preds, loss, acc

    # def build_metrics(self):
    #     labels = tf.argmax(self.model.y, -1)
    #     predictions = tf.argmax(self.model.predictions)

    #     with tf.variable_scope("metrics"):

    #         acc, acc_op = tf.metrics.accuracy(labels, predictions)
    #         auc, auc_op = tf.metrics.auc(labels, predictions)
    #         prec, prec_op = tf.metrics.precision(tf.one_hot(labels, 1), tf.one_hot(predictions, 1))
    #         recall, recall_op = tf.metrics.recall(tf.one_hot(labels, 1), tf.one_hot(predictions, 1))

    #         per_class = []
    #         per_class_op = []
    #         for k in range(5):
    #             prec, prec_op = tf.metrics.precision(labels=tf.equal(labels, k), predictions=tf.equal(predictions, k))
    #             rec, rec_op = tf.metrics.recall(labels=tf.equal(labels, k), predictions=tf.equal(predictions, k))
    #             per_class.append([prec, rec])
    #             per_class_op.append([prec_op, rec_op])


    #         self.metrics = [acc, auc, prec, recall, per_class]
    #         self.update_ops = [acc_op, auc_op, prec_op, recall_op, per_class_op]

    #     self.reset_metrics = tf.variables_initializer(
    #         tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics"))


    def validate(self, cur_it):
        losses = []
        accs = []
        labels = np.array([], dtype=np.int32)
        predictions = np.array([], dtype=np.int32)

        self.sess.run(self.data.validation_iterator.initializer)
        # self.sess.run(self.reset_metrics)

        i = 0
        while True:

            try:
                validation_x, validation_y = self.sess.run(self.data.validation_next)
                feed_dict = {self.model.x: validation_x, self.model.y: validation_y, self.model.is_training: False}
                label, prediction, loss, acc = self.sess.run([self.model.labels, self.model.predictions, self.model.loss, self.model.accuracy], feed_dict=feed_dict)
                
                losses.append(loss)
                accs.append(acc)

                labels = np.append(labels, label)
                predictions = np.append(predictions, prediction)
            
                i += 1
            except tf.errors.OutOfRangeError:
                break

        loss, acc = np.mean(losses), np.mean(accs)
        
        summaries_dict = {
            'loss': loss,
            'acc': acc,
        }

        class_metrics, confusion_m, avg_acc = self.per_class_metrics(labels, predictions)
        print("============================================")
        print("VALIDATION")
        print("============================================")
        print("LOSS: {}".format(loss))
        print("ACCURACY: {}".format(acc))
        print("AVG_ACC_PER_CLASS: {}".format(avg_acc))
        print("PER_CLASS_METRICS:")
        print(class_metrics.draw())
        print("CONFUSION_MATRIX:")
        print(confusion_m.draw())

        self.logger.summarize(cur_it, summaries_dict=summaries_dict, summarizer="test")

    def per_class_metrics(self, labels, predictions):

        _, counts = np.unique(labels, return_counts=True)
        precision, recall, _, _ = score(labels, predictions)
        C = confusion_matrix(labels, predictions)
        avg_acc_per_class = np.average(recall)

        t = Texttable()
        t.add_rows([
            ['Metric', 'CAR', 'BUS', 'TRUCK', 'OTHER'],
            ['Count labels'] + counts.tolist(),
            ['Precision'] + precision.tolist(),
            ['Recall'] + recall.tolist()
        ])

        t2 = Texttable()
        t2.add_rows([
            ['-', 'CAR', 'BUS', 'TRUCK', 'OTHER'],
            ['CAR'] + C[0].tolist(),
            ['BUS'] + C[1].tolist(),
            ['TRUCK'] + C[2].tolist(),
            ['OTHER'] + C[3].tolist()
        ])

        return t, t2, avg_acc_per_class