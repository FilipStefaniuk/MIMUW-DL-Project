import tensorflow as tf
import os


class DataGenerator:
    def __init__(self, config):
        self.config = config
        self._build_dataset()

    def _parse_example(self, example_proto):
        features = {
            "image/encoded": tf.FixedLenFeature([], tf.string),
            "image/height": tf.FixedLenFeature([], tf.int64),
            "image/width": tf.FixedLenFeature([], tf.int64),
            "image/filename": tf.FixedLenFeature([], tf.string),
            "image/class/label": tf.FixedLenFeature([], tf.int64)}
        parsed_features = tf.parse_single_example(example_proto, features)
        return parsed_features

    def _filter_classes(self, parsed_features):
        label = parsed_features["image/class/label"]
        result = True

        if "CAR" in self.config.remove_class:
            result = tf.logical_and(result, tf.not_equal(label, 0))

        if "BUS" in self.config.remove_class:
            result = tf.logical_and(result, tf.not_equal(label, 1))

        if "TRUCK" in self.config.remove_class:
            result = tf.logical_and(result, tf.not_equal(label, 2))

        if "OTHER" in self.config.remove_class:
            result = tf.logical_and(result, tf.not_equal(label, 3))

        return result

    def _remap_label(self, label):
        result = label

        if "CAR" in self.config.remove_class:
            result -= 1

        if "BUS" in self.config.remove_class:
            result -= tf.cast(label > 1, tf.int64)

        if "TRUCK" in self.config.remove_class:
            result -= tf.cast(label > 2, tf.int64)

        return result

    def _parse_image(self, parsed_features):
        image = tf.image.decode_jpeg(parsed_features["image/encoded"], channels=3)
        image = tf.image.resize_images(
            image, 
            size=(self.config.input_size, self.config.input_size), 
            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        )

        label = parsed_features["image/class/label"]
        label = self._remap_label(label)
        label = tf.one_hot(label, depth=(4 - len(self.config.remove_class)))

        return image, label

    def _read_dataset(self, filenames):
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self._parse_example, num_parallel_calls=self.config.num_parallel_threads)
        dataset = dataset.filter(self._filter_classes)
        dataset = dataset.map(self._parse_image, num_parallel_calls=self.config.num_parallel_threads)
        return dataset

    def _train_augmentation(self, image, label):

        if "HORIZONTAL_FLIP" in self.config.augmentation:
            image = tf.image.random_flip_left_right(image)

        return image, label

    def _validation_augmentation(self, image, label):

        result = [tf.expand_dims(image, 0)]

        if self.config.validation_augmentation:

            if "HORIZONTAL_FLIP" in self.config.augmentation:
                result.append(tf.expand_dims(tf.image.flip_left_right(image), 0))

        return tf.concat(result, 0), label

    def _build_dataset(self):

        filenames = os.listdir(self.config.data_dir)
        test_filenames = [os.path.join(self.config.data_dir, file) for file in filenames if file.startswith('test')]
        validation_filenames = [os.path.join(self.config.data_dir, file) for file in filenames if file.startswith('validation')]
        train_filenames = [os.path.join(self.config.data_dir, file) for file in filenames if file.startswith('train')]

        assert test_filenames
        assert validation_filenames
        assert train_filenames

        self.train = self._read_dataset(train_filenames)
        self.train = self.train.apply(tf.contrib.data.shuffle_and_repeat(2 * self.config.batch_size, None))
        self.train = self.train.map(self._train_augmentation)
        self.train = self.train.batch(self.config.batch_size).prefetch(1)

        self.train_next_batch = self.train.make_one_shot_iterator().get_next()

        self.validation = self._read_dataset(validation_filenames)
        self.validation = self.validation.batch(self.config.batch_size)
        self.validation = self.validation.map(self._validation_augmentation, num_parallel_calls=self.config.num_parallel_threads)

        self.validation_iterator = self.validation.make_initializable_iterator()
        self.validation_next_batch = self.validation_iterator.get_next()
        
        self.test = self._read_dataset(test_filenames)
        self.test = self.test.batch(self.config.batch_size)

        self.test_iterator = self.test.make_initializable_iterator()
        self.test_next_batch = self.test_iterator.get_next()
