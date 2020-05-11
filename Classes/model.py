from __future__ import absolute_import

import datetime
import math
import os
import random
import sys
import json
import neural_structured_learning as nsl
import numpy as np
import split_folders
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import plotly
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from augmentors import *
from PIL import Image
from sklearn import preprocessing
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import DenseNet121, preprocess_input
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


config = tf.compat.v1.ConfigProto(gpu_options =
                         tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
# device_count = {'GPU': 1}
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

class DL():

    available_augmentations = {'erase' : RandomErasing(), 'blend' : MixingImages()}

    def __init__(self, config):
        self.data_dir = config['data_dir']
        self.results_dir = config['results_dir']
        self.model_dir = config['model_dir']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']
        self.target_size = config['target_size']
        self.metrics = config['metrics']
        self.additional_augmentations = config['additional_augmentations']
        self.channels = 3
        self.train_set_size = .7
        self.val_set_size = .15
        self.test_set_size = .15
        self.weights_filename = "final_model" + "_weights.h5"
        self.train_data_generator = ImageDataGenerator(
                preprocessing_function=preprocess_input
               )
        self.test_data_generator = ImageDataGenerator(
            preprocessing_function=preprocess_input
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
        total = 0
        for root, dirs, files in os.walk(self.data_dir):
            total += len(files)

        self.n_samples = total

    def set_real_labels(self):
        """Finds the real labels by looking at the subdirectories in de data directory.
            Assumes that subdirectories are called according to the classes
        """
        self.real_labels = [d for d in os.listdir(self.data_dir + '/train') if os.path.isdir(os.path.join(self.data_dir + '/train', d))]

    def count_labels(self):
        """Counts the amount of labels in the data
        """
        self.num_classes = len(self.real_labels)

    def set_image_shape(self):
        """Set the shape of the images
        """
        self.input_shape = (self.target_size[0], self.target_size[1], self.channels)

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
        self.n_training_samples = int(self.train_set_size * self.n_samples)

    def compute_n_validation_samples(self):
        """Computes the amount of validation samples
        """
        self.n_validation_samples = int(self.val_set_size * self.n_samples)

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
        self.train_gen = self.train_data_generator.flow_from_directory(
            self.data_dir + '/train',
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical'
        )

    def set_validation_data_generator(self):
        """Create the generator for the validation data
        """
        self.val_gen = self.test_data_generator.flow_from_directory(
            self.data_dir + '/val',
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False

        )

    def set_test_data_generator(self):
        """Create the generator for the training data
        """
        self.test_gen = self.test_data_generator.flow_from_directory(
            self.data_dir + '/test',
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False

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

    def wrap_generator(self, batches):
        """Take as input a Keras ImageGen (Iterator) and generate more augmentations
        according to augmentor functions.
        """
        while True:
            batch_x, batch_y = next(batches)

            # check if there are any additional augmentations to apply
            if self.additional_augmentations:
                # create an empty batch
                batch_augmented = np.zeros((batch_x.shape[0], self.input_shape[0], self.input_shape[1], self.channels))
                # get all augmentor objects
                augmentors = [augmentor for augmentation_type, augmentor in available_augmentations if augmentation_type in self.additional_augmentations]
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
        # callbacks = self.get_callbacks(patience_lr=10)

        train_generator = self.wrap_generator(self.train_gen)
        val_generator = self.wrap_generator(self.val_gen)

        self.model.fit_generator(
                        train_generator,
                        steps_per_epoch = int(math.ceil(self.n_training_samples // self.batch_size)),
                        epochs=self.epochs,
                        shuffle=True,
                        verbose=1,
                        validation_data = val_generator,
                        validation_steps = int(math.ceil(self.n_validation_samples // self.batch_size)),
                        )

    def evaluate_model(self):
        """Evaluate the model
        """
        Y_pred = self.model.predict(self.test_gen)
        y_pred = np.argmax(Y_pred, axis=1)

        accuracy = accuracy_score(self.test_gen.classes, y_pred)
        precision = precision_score(self.test_gen.classes, y_pred, average='weighted')
        recall = recall_score(self.test_gen.classes, y_pred, average='weighted')
        f1 = f1_score(self.test_gen.classes, y_pred, average='weighted')

        print("Accuracy in test set: %0.1f%% " % (accuracy * 100))
        print("Precision in test set: %0.1f%% " % (precision * 100))
        print("Recall in test set: %0.1f%% " % (recall * 100))
        print("F1 score in test set: %0.1f%% " % (f1 * 100))

        results = pd.DataFrame(columns=['Accuracy', 'Precision', 'Recall', 'F1'])
        results['Accuracy'] = [accuracy]
        results['Precision'] = [precision]
        results['Recall'] = [recall]
        results['F1'] = [f1]

        results.to_latex(self.results_dir + '/scores.tex')
        results.to_excel(self.results_dir + '/scores.xlsx')

    def save_results(self):
        # save the model
        self.model.save(self.model_dir)
        # Get the dictionary containing each metric and the loss for each epoch
        history_dict = self.model.history.history

        # list of epoch ints
        n_epochs = [i + 1 for i in range(self.epochs)]

        # make plot of the loss
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=n_epochs, y=history_dict['loss'],
                            mode='lines',
                            name='loss')   )
        fig.add_trace(go.Scatter(x=n_epochs, y=history_dict['val_loss'],
                            mode='lines',
                            name='val loss'))

        fig.update_layout(
                title="Loss and validation loss",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                font=dict(
                    family="Helvetica",
                    size=18,
                    color="#7f7f7f"
                )
            )

        plotly.offline.plot(fig, filename = self.results_dir + '/loss_plot.html', auto_open=False)
        # fig.to_html(self.results_dir + '/loss_plot.html')
        fig.write_image(format='png', file=self.results_dir + '/loss_plot.png')


        # make plots of all the history elements
        for metric in self.metrics:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=n_epochs, y=history_dict[metric],
                                mode='lines',
                                name=metric))
            fig.add_trace(go.Scatter(x=n_epochs, y=history_dict['val_' + metric],
                                mode='lines',
                                name='val ' + metric))

            fig.update_layout(
                    title="Loss and validation loss",
                    xaxis_title="Epoch",
                    yaxis_title=metric,
                    font=dict(
                        family="Helvetica",
                        size=18,
                        color="#7f7f7f"
                    )
                )

            # fig.to_html(self.results_dir + '/' + metric + '_plot.html')
            plotly.offline.plot(fig, filename = self.results_dir + '/' + metric + '_plot.html', auto_open=False)
            fig.write_image(format='png', file=self.results_dir + '/' + metric + '_plot.png')


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
                        metrics=self.metrics)
        self.model = custom_model

