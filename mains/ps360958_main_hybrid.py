import tensorflow as tf

from data_loader.data_generator import DataGenerator
from models.example_model import ExampleModel
from trainers.simple_trainer import SimpleTrainer
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
from models.ps360958_hybrid_sum import MyModel


def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create an instance of the model you want
    model = MyModel(config)
    # create your data generator
    data = DataGenerator(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = SimpleTrainer(sess, model, data, config, logger)
    saverExternal = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='external'))
    saverExternal.restore(sess, "experiments/model_8/model8.ckpt_2")
    saver5 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ps360958_5_part'))
    saver5.restore(sess, "experiments/ps360958_5_part/checkpoint/-2320")
    #saver5 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ps360958_5_part'))
    #saver5.restore(sess, "experiments/ps360958_5_part_backup/checkpoint/-1600")
    saver6 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ps360958_6_part'))
    saver6.restore(sess, "experiments/ps360958_6_part/checkpoint/-2400")
    saver8 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ps360958_8_part'))
    saver8.restore(sess, "experiments/ps360958_8_part/checkpoint/-5600")
    # here you train your model
    #trainer.train()
    trainer.validate()


if __name__ == '__main__':
    main()
