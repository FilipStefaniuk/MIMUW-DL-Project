import tensorflow as tf
import os


class DataGenerator:
    def __init__(self, config):
        batch_size = config['batch_size']
        directory = config['data_dir']
        self.num_parallel_threads = config['num_parallel_threads']
        filenames = os.listdir(directory)
        test_filenames = [os.path.join(directory, file) for file in filenames if file.startswith('test')]
        validation_filenames = [os.path.join(directory, file) for file in filenames if file.startswith('validation')]
        train_filenames = [os.path.join(directory, file) for file in filenames if file.startswith('train')]
        assert test_filenames
        assert validation_filenames
        assert train_filenames
        self.test = self._read_dataset(test_filenames)
        self.validation_iterator = self._read_dataset(validation_filenames).batch(batch_size).make_initializable_iterator()
        self.validation_next = self.validation_iterator.get_next()
        self.train = self._read_dataset(train_filenames)
        batch_train = self.train.apply(tf.contrib.data.shuffle_and_repeat(2 * batch_size, None))
        batch_train = batch_train.batch(batch_size).prefetch(1)
        self.next_batch_node = batch_train.make_one_shot_iterator().get_next()
        self.config = config

    def next_batch(self, unused_argument=None):
        return self.next_batch_node

    def augment(self, image):
        image = tf.image.resize_images(image, size=(256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return image
        # data = tf.concat([image, label], axis=2)
        # data = tf.image.resize_images(data, size=(256, 256), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # image = data[:, :, :-1]
        # label = data[:, :, -1:]
        # return image, label

    def _parse_image(self, example_proto):
        features = {
            "image/encoded": tf.FixedLenFeature([], tf.string),
            "image/height": tf.FixedLenFeature([], tf.int64),
            "image/width": tf.FixedLenFeature([], tf.int64),
            "image/filename": tf.FixedLenFeature([], tf.string),
            "image/class/label": tf.FixedLenFeature([], tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, features)
        image = tf.image.decode_jpeg(parsed_features["image/encoded"], channels=3)
        label = parsed_features["image/class/label"]
        image = self.augment(image)
        label = tf.one_hot(label, depth=4)
        return image, label

    def _read_dataset(self, filenames):
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self._parse_image, num_parallel_calls=self.num_parallel_threads)
        return dataset


# sess = tf.InteractiveSession()
# mopset = DataGenerator(None)
# iterator = mopset.test.make_one_shot_iterator()
# elem = iterator.get_next()
# print(elem[0].eval())

# with tf.Session() as sess:
#     mopset = DataGenerator(None)
#     for _ in range(100):
#         batch = mopset.next_batch()
#         print(batch[0].eval().shape)
