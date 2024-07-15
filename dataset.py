import tensorflow as tf
from tensorflow.keras import models, layers
# import matplotlib.pyplot as plt
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Your TensorFlow code


class Dataset:

    IMG_SIZE = 256
    BATCH_SIZE = 32
    CHANNELS = 3 # rgb color
    EPOCHS = 50

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "PlantVillage",
        shuffle=True,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    class_names = dataset.class_names
    print(class_names)

    def get_dataset(self):
        return self.dataset

    def get_imgsize(self):
        return self.IMG_SIZE

    def get_batch_size(self):
        return self.BATCH_SIZE

    def get_channels(self):
        return self.CHANNELS

    def get_epochs(self):
        return self.EPOCHS

    @staticmethod
    def get_dataset_partitions_tf(dataset, train_split=0.8, val_split=0.1,
                                  test_split=0.1, shuffle=True, shuffle_size=10000):

        # length of datasize
        ds_size = len(dataset)
        # train size 80 %
        if shuffle:
            dataset = dataset.shuffle(shuffle_size, seed=12)

        train_size = int(train_split * ds_size)
        # validation size 10%
        val_size = int(val_split * ds_size)
        # test size remaining data

        trn_ds = dataset.take(train_size)
        vl_ds = dataset.skip(train_size).take(val_size)
        tst_ds = dataset.skip(train_size).skip(val_size)

        return trn_ds, vl_ds, tst_ds

    def prepare_data(self):
        trn_ds, vl_ds, tst_ds = self.get_dataset_partitions_tf(self.get_dataset())

        print(len(trn_ds))
        print(len(vl_ds))
        print(len(tst_ds))

        # improving performance
        # optimized
        trn_ds = trn_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        vl_ds = vl_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
        tst_ds = tst_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

        return trn_ds, vl_ds, tst_ds

    # before image processing we need to scale
    @staticmethod
    def get_preprocessing_layers(img_size):
        re_size_and_rescale = tf.keras.Sequential([
            layers.Resizing(img_size, img_size),
            layers.Rescaling(1.0 / 255)
        ])

        data_augment = tf.keras.Sequential([
            layers.RandomFlip("horizontal_and_vertical"),
            layers.RandomRotation(0.2),
        ])

        return re_size_and_rescale, data_augment


# Example usage
dataset_instance = Dataset()
train_ds, val_ds, test_ds = dataset_instance.prepare_data()
resize_and_rescale, data_augmentation = Dataset.get_preprocessing_layers(dataset_instance.get_imgsize())

    # summary we loaded our data in tensorflow data set
    # visualisation not done
    # train test split
    # implemented some layers of preprocessing
    # we will be using these layers to train our model

