from base_classes.base_train import BaseTrain
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score
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
            label, prediction, loss = self.train_step()
            losses.append(loss)
            accs.append(accuracy_score(np.argmax(label, 1), np.argmax(prediction, 1)))

        loss = np.mean(losses)
        acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        
        print("============================================")
        print("TRAINING")
        print("============================================")
        print("GLOBAL_STEP {}".format(cur_it))
        print("LOSS: {}".format(loss))
        print("ACCURACY: {0}".format(acc))

        train_summaries_dict = {
            'loss': loss,
            'acc': acc,
        }

        self.logger.summarize(step=cur_it, summarizer="train", summaries_dict=train_summaries_dict)
        
        if self.config.validate:
            self.validate()
        
        if self.config.save_model:
            self.model.save(self.sess)

    def train_step(self):
        batch_x, batch_y = self.sess.run(self.data.train_next_batch)
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, labels, preds, loss = self.sess.run([self.model.train_step, self.model.labels, self.model.predictions, self.model.loss],
                                     feed_dict=feed_dict)
        return labels, preds, loss

    def validate(self):

        cur_it = self.model.global_step_tensor.eval(self.sess)
        self.sess.run(self.data.validation_iterator.initializer)

        labels = []
        predictions = []

        i = 0
        while True:

            try:
                data_x, data_y = self.sess.run(self.data.validation_next_batch)

                prediction = np.zeros(data_y.shape)
                for j in range(data_x.shape[0]):
                    feed_dict = {self.model.x: data_x[j], self.model.is_training: False}
                    prediction += self.sess.run(self.model.predictions, feed_dict=feed_dict)
                
                predictions.append(np.argmax(prediction, 1))
                labels.append(np.argmax(data_y, 1))

                i += 1
            
                # if i > 10:
                    # break

            except tf.errors.OutOfRangeError:
                break

        labels = np.concatenate(labels)
        predictions = np.concatenate(predictions)
        metrics = self._compute_metrics(labels, predictions)

        t1 = self._make_metric_table(metrics)
        t2 = self._make_confusion_table(labels, predictions)

        print("============================================")
        print("VALIDATION")
        print("============================================")
        print("GLOBAL_STEP: {}".format(cur_it))
        print("ACCURACY: {}".format(metrics['ACC']))
        print("AVG_ACC_PER_CLASS: {}".format(metrics['AVG_ACC']))
        print("PER_CLASS_METRICS:")
        print(t1.draw())
        print("CONFUSION_MATRIX:")
        print(t2.draw())

        header = self._make_classes_header()
        val_summaries_dict = {}
        val_summaries_dict.update({'COUNT_' + h: v for h, v in zip(header, metrics['COUNT'])})
        val_summaries_dict.update({'PRECISION_' + h: v for h, v in zip(header, metrics['PRECISION'])})
        val_summaries_dict.update({'RECALL_' + h: v for h, v in zip(header, metrics['RECALL'])})
        val_summaries_dict.update({'F1_SCORE_' + h: v for h, v in zip(header, metrics['F1_SCORE'])})
        val_summaries_dict.update({'GLOBAL_STEP': np.int64(cur_it), 'ACCURACY': metrics['ACC'], 'AVG_ACC_PER_CLASS': metrics['AVG_ACC']})
        self.logger.summarize(step=cur_it, summarizer="test", summaries_dict=val_summaries_dict)
        # self.logger.summarize(cur_it, summaries_dict=summaries_dict, summarizer="test")

    def _compute_metrics(self, labels, predictions):
        _, counts = np.unique(labels, return_counts=True)
        acc = accuracy_score(labels, predictions)
        prec, rec, f1, _ = score(labels, predictions)

        return {
            'COUNT': counts,
            'ACC': acc,
            'AVG_ACC': np.average(rec),
            'PRECISION': prec,
            'RECALL': rec,
            'F1_SCORE': f1
        }
    
    def _make_classes_header(self):
        header = []
        if 'CAR' not in self.config.remove_class:
            header.append('CAR')
        if 'BUS' not in self.config.remove_class:
            header.append('BUS')
        if 'TRUCK' not in self.config.remove_class:
            header.append('TRUCK')
        if 'OTHER' not in self.config.remove_class:
            header.append('OTHER')
        return header

    def _make_metric_table(self, metrics):
        header = ['Metric']
        header += self._make_classes_header()

        t = Texttable()
        t.add_rows([
            header,
            ['Count labels'] + metrics['COUNT'].tolist(),
            ['Precision'] + metrics['PRECISION'].tolist(),
            ['Recall'] + metrics['RECALL'].tolist(),
            ['F1 score'] + metrics['F1_SCORE'].tolist()
        ])

        return t

    def _make_confusion_table(self, labels, predictions):
        header = self._make_classes_header()
        C = confusion_matrix(labels, predictions)
        t = Texttable()
        t.add_row(['-'] + header)
        for i, h in enumerate(header):
            t.add_row([h] + C[i].tolist())
        return t