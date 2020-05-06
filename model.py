from __future__ import absolute_import

import datetime
import math
import os
import random

import neural_structured_learning as nsl
import numpy as np
import split_folders
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from augmentors import *
from PIL import Image
from sklearn import preprocessing
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DL():

    self.available_augmentations = {'erase' : RandomErasing(), 'blend' : MixingImages()}

    def __init__(self, data_dir, epochs, bs, lr, target_size):
        self.data_dir = data_dir
        self.epochs = epochs
        self.batch_size = bs
        self.learning_rate = lr
        self.target_size = target_size
        self.additional_augmentations = []
        self.channels = 3
        self.train_set_size = .7
        self.val_set_size = .15
        self.test_set_size = .15
        self.weights_filename = "final_model" + "_weights.h5"
        self.data_generator = ImageDataGenerator(
                rescale=1./255,
               )

        # compute some properties of the data
        self.compute_n_samples()
        self.set_real_labels()
        self.count_labels()

        # set image shape
        self.set_image_shape()

        # compute the amount of samples in each set
        self.compute_n_training_samples()
        self.compute_n_validation_samples()

        # set the generators
        self.set_training_data_generator()
        self.set_validation_data_generator()
        self.set_test_data_generator()

        # build the base model
        self.build_base_model()

    def compute_n_samples(self):
        """Computes the amount of samples in the data directory
        """
        files = os.listdir(self.data_dir) # dir is your directory path
        self.n_samples = len(files)

    def set_real_labels(self):
        """Finds the real labels by looking at the subdirectories in de data directory.
            Assumes that subdirectories are called according to the classes
        """
        self.real_labels = [d for d in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, d))]

    def count_labels(self):
        """Counts the amount of labels in the data
        """
        self.num_classes = len(self.real_labels)

    def set_image_shape(self):
        """Set the shape of the images
        """
        self.input_shape = (self.target_size[0], self.target_size[1], self.target_size[2])

    def add_additional_augmentations(self, augmentations):
        """Add one or more of the available augmentations ('erase' and 'blend')

        Arguments:
            augmentations {list} -- with string of the augmentation type to add
        """
        self.additional_augmentations = augmentations

    def set_image_channels(self, channels):
        """Set the amount of channels of the input images

        Arguments:
            channels {int} -- amount of channels
        """
        self.channels = channels

    def set_weightsfile(self, filename):
        """Set a custom path to save the model weights

        Arguments:
            filename {path} -- filepath
        """
        self.weights_filename = filename

    def set_train_set_size(self, fraction):
        """Sets the size of the training set

        Arguments:
            fraction {float} -- the fraction of total data to be used as the training set
        """
        self.train_set_size = fraction

    def set_validation_set_size(self, fraction):
        """Sets the size of the validation set

        Arguments:
            fraction {float} -- the fraction of total data to be used as the validation set
        """
        self.val_set_size = fraction

    def set_test_set_size(self, fraction):
        """Sets the size of the test set

        Arguments:
            fraction {float} -- the fraction of total data to be used as the test set
        """
        self.test_set_size = fraction

    def compute_n_training_samples(self):
        """Computes the amount of training samples
        """
        self.n_training_samples = self.train_set_size * self.n_samples

    def compute_n_validation_samples(self):
        """Computes the amount of validation samples
        """
        self.n_validation_samples = self.val_set_size * self.n_samples

    def set_data_generator(self, generator):
        """Change the data generator

        Arguments:
            generator {tensorflow data generator} -- data generator
        """
        self.data_generator = generator
        # reset the generators
        self.set_training_data_generator()
        self.set_validation_data_generator()
        self.set_test_data_generator()

    def set_training_data_generator(self):
        """Create the generator for the training data
        """
        self.train_gen = self.data_generator.flow_from_directory(
            self.data_dir + '/train',
            target_size=self.target_size,
            batch_size=self.batch_size
        )

    def set_validation_data_generator(self):
        """Create the generator for the validation data
        """
        self.val_gen = self.data_generator.flow_from_directory(
            self.data_dir + '/val',
            target_size=self.target_size,
            batch_size=self.batch_size
        )

    def set_test_data_generator(self):
        """Create the generator for the training data
        """
        self.test_gen = self.data_generator.flow_from_directory(
            self.data_dir + '/test',
            target_size=self.target_size,
            batch_size=self.batch_size
        )

    def set_model(self, model):
        """Set a custom tensorflow keras model. Must be compiled before calling this function

        Arguments:
            model {a compiled tensorflow keras model}
        """
        self.model = model

    @staticmethod
    def split_directory_into_train_val_test_sets(indir, outdir, ratios):
        """Split your dataset directory into three separate directories for train, validation, and test sets

        Arguments:
            indir {/yourpath} -- path to dataset directory
            outdir {'/yourpath'} -- path to new directory
            ratios {tuple of floats that sum to 1} -- in the form (train_size, val_size, test_size)
        """
        split_folders.ratio(indir, output=outdir, seed=1337, ratio=ratios) 

    @staticmethod
    def get_callbacks(name_weights, patience_lr):
        mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
        reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=patience_lr, verbose=1, min_delta=1e-4, mode='min')
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        return [mcp_save, reduce_lr_loss]

    def wrap_generator(self, batches):
        """Take as input a Keras ImageGen (Iterator) and generate more augmentations
        according to augmentor functions.
        """
        while True:
            batch_x, batch_y = next(batches)

            # check if there are any additional augmentations to apply
            if self.augmentations:
                # create an empty batch
                batch_augmented = np.zeros((batch_x.shape[0], self.input_shape[0], self.input_shape[1], self.channels))
                # get all augmentor objects
                augmentors = [augmentor for augmentation_type, augmentor in self.available_augmentations if augmentation_type in self.additional_augmentations]
                # possibly apply augmentation to each image in batch
                for i in range(batch_x.shape[0]):
                    img_to_augment = batch_x[i]
                    # apply each of the additional augmentations
                    for augmentor in augmentors:
                        img_to_augment = augmentor(img_to_augment)
                    batch_augmented[i] = img_to_augment
                yield(batch_augmented, batch_y)
            else:
                yield (batch_x, batch_y)

    def train_model(self):
        """Train the model
        """
        callbacks = get_callbacks(name_weights = self.weights_filename, patience_lr=10)

        train_generator = self.wrap_generator(self.train_gen)
        val_generator = self.wrap_generator(self.val_gen)

        self. model.fit(
                        train_generator,
                        steps_per_epoch = int(self.n_training_samples // self.batch_size),
                        epochs=self.epochs,
                        shuffle=True,
                        verbose=1,
                        validation_data = val_generator,
                        validation_steps = int(self.n_validation_samples // self.batch),
                        callbacks = callbacks)

    def evaluate_model(self):
        filenames = self.test_gen.filenames
        nb_samples = len(filenames)

        predictions = self.model.predict(test_gen, steps = nb_samples)
        true_labels = self.test_gen.classes

        y_true = true_labels
        y_pred = np.array([np.argmax(x) for x in predictions])

        test_acc = sum(y_true == y_pred) / len(y_true)
        print('Accuracy: {}'.format(test_acc))
        self.test_acc = test_acc


    def build_base_model(self):
        # build the Densenet network
        densenet = DenseNet121(weights='imagenet',
                                include_top=False,
                                input_shape=self.input_shape)

        for layer in densenet.layers:
            layer.trainable = False

        # make batch normalization layers trainable to prevent overfitting
        for layer in densenet.layers:
            if "BatchNormalization" in layer.__class__.__name__:
                layer.trainable = True

        x = densenet.output
        x = Flatten()(x)

        layer_units = (64, 16)
        for num_units in layer_units:
            x = Dense(num_units, activation='relu')(x)
            x = Dropout(0.4)(x)

        predictions = Dense(self.num_classes, activation='softmax')(x)
        custom_model = Model(inputs=densenet.input, outputs=predictions)
        optimizer = optimizers.Adam(lr=self.learning_rate)
        custom_model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                        metrics=['acc', 'AUC', 'Precision', 'Recall'])
        self.model = custom_model

if __name__== '__main__':
    data_dir = 'C:/Users/trist/Documents/GitHub/DeepLearningProject/dataset-split'
    epochs = 20
    batch_size = 16
    learning_rate = 0.0001
    target_size = (256, 256)

    session = DL(data_dir, epochs, batch_size, learning_rate, target_size)
    session.train_model()
    session.evaluate_model()

