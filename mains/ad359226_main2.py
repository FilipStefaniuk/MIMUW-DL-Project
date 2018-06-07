import tensorflow as tf

from data_loader.ad359226_data_generator import DataGenerator
from models.example_model import ExampleModel
from trainers.ad359226_trainer import SimpleTrainer 
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
from models.ad359226_2 import MyModel


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

    # here you train your model
    trainer.train()
    trainer.validate()


if __name__ == '__main__':
    main()
