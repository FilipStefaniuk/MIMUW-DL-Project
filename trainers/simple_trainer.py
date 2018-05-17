from base_classes.base_train import BaseTrain
from tqdm import tqdm
import tensorflow as tf
import numpy as np


class SimpleTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(SimpleTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):

        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        epoch_str = 'Epoch {0}/{1}'.format(cur_epoch, self.config.num_epochs)
        loop = tqdm(range(self.config.num_iter_per_epoch), ncols=120, desc=epoch_str)
        
        losses = []
        accs = []
        
        for _ in loop:
            loss, acc = self.train_step()
            losses.append(loss)
            accs.append(acc)

        loss = np.mean(losses)
        acc = np.mean(accs)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        
        print('Global step: {}'.format(cur_it))
        print('Training: loss {0}, accuracy {1}'.format(loss, acc))

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
        _, loss, acc = self.sess.run([self.model.train_step, self.model.loss, self.model.accuracy],
                                     feed_dict=feed_dict)
        return loss, acc

    def validate(self, cur_it):
        losses = []
        accs = []

        self.sess.run(self.data.validation_iterator.initializer)

        while True:

            try:
                validation_x, validation_y = self.sess.run(self.data.validation_next)
                feed_dict = {self.model.x: validation_x, self.model.y: validation_y, self.model.is_training: False}
                loss, acc = self.sess.run([self.model.loss, self.model.accuracy], feed_dict=feed_dict)
                
                losses.append(loss)
                accs.append(acc)
            
            except tf.errors.OutOfRangeError:
                break

        loss, acc = np.mean(losses), np.mean(accs)
        
        summaries_dict = {
            'loss': loss,
            'acc': acc,
        }

        print('Test: loss {0}, accuracy {1}'.format(loss, acc))
        self.logger.summarize(cur_it, summaries_dict=summaries_dict, summarizer="test")