from __future__ import print_function, division
import os
# from tensorflow import keras
# ### hack tf-keras to appear as top level keras
# import sys
# sys.modules['keras'] = keras
### end of hack

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, Conv2DTranspose, LeakyReLU, UpSampling2D, Conv2D, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
# from tensorflow. keras.layers.advanced_activations import LeakyReLU
# from tensorflow.keras.layers.convolutional import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import os
from PIL import Image
import numpy as np
import sys
from keras.utils import to_categorical, plot_model
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.initializers import RandomNormal
from keras.preprocessing.image import array_to_img, img_to_array
from numpy.random import randint

class CGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 6
        self.latent_dim = 100
        self.batch_size = 32
        self.real_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator = self.build_generator()

        noise = Input(shape=(self.latent_dim, ))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # during generator updating,  the discriminator is fixed (will not be updated).
        self.discriminator.trainable = False

        # The discriminator takes generated image and label as input and determines its validity
        validity = self.discriminator([img, label])

        self.cgan_model = Model([noise, label], validity)
        self.cgan_model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                                optimizer=optimizer,
                                metrics=['accuracy'])

    def build_generator(self):

        image_resize = self.img_rows // 4

        inputs = Input(shape=(self.latent_dim,), name='z_input')
        labels = Input(shape=(1,), name='class_labels')

        embedded_label = Embedding(self.num_classes, 50)(labels)
        n_nodes = image_resize * image_resize
        embedded_label = Dense(n_nodes)(embedded_label)
        embedded_label = Reshape((image_resize, image_resize, 1))(embedded_label)

        n_nodes = 128 * image_resize * image_resize
        gen = Dense(n_nodes)(inputs)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Reshape((image_resize, image_resize, 128))(gen)

        x = Concatenate()([gen, embedded_label])

        x = Conv2DTranspose(filters=128, kernel_size=4, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)

        x = Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)

        x = Conv2DTranspose(filters=32, kernel_size=4, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2DTranspose(filters=self.channels, kernel_size=4, strides=1, padding='same')(x)
        x = Activation('tanh')(x)
        # input is conditioned by labels
        generator = Model([inputs, labels], x, name='generator')
        generator.summary()
        return generator

    def build_discriminator(self):

        model_input = Input(shape=self.img_shape, name='discriminator_input')
        x = model_input
        labels = Input(shape=(1,))

        label_embedded = Embedding(self.num_classes, 50)(labels)
        n_nodes = self.img_shape[0] * self.img_shape[1] * self.img_shape[2]
        label_embedded = Dense(n_nodes)(label_embedded)
        labels_embedded = Reshape((self.img_shape[0], self.img_shape[1], 3))(label_embedded)

        x = Concatenate()([x, labels_embedded])

        x = Conv2D(filters=32, kernel_size=4, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(filters=64, kernel_size=4, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(filters=128, kernel_size=4, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Conv2D(filters=256, kernel_size=4, strides=1, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(1)(x)
        # x = Activation('sigmoid')(x)

        # model_input is conditioned by labels
        discriminator = Model([model_input, labels], x, name='discriminator')
        discriminator.summary()
        return discriminator


    def check_string_in_list(self, match_list, string):
        """Return matching string fro   m a list of possible matches"""

        # for possible_match in match_list:
        for possible_match, label in match_list.items():
            if possible_match in string:
                return label

        return 'No matching label in list of possible matches'

    def load_images_and_labels(self, rootdir, possible_labels):
        """

        Load and shuffle   the images and the labels from a directory. Assumes labels are given in the filenames.

        rootdir (str) : the directory where the images are stored
        possible_labels (list) : a list containing the possible labels of the task

        """
        loaded_images = list()
        labels = list()

        for subdir, dirs, files in os.walk(rootdir):
            for filename in files:
                image = Image.open(subdir + '/' + filename)#.convert('L')
                newsize = (self.img_cols, self.img_rows) 
                im1 = image.resize(newsize)
                pixels = img_to_array(im1)

                # store loaded image
                loaded_images.append(pixels)
                # find label in filename and store label
                labels.append(self.check_string_in_list(possible_labels, filename))

        labels = to_categorical(labels)

        # print ('Loaded {} images succesfully with {} unique classes'.format(len(loaded_images), len(set(list(labels)))))
        return np.asarray(loaded_images), labels

    def generate_noise(self, type_of_noise, size):
        if type_of_noise == "normal_noise":
            return np.random.normal(0, 1, size=[size, self.latent_dim])

        elif type_of_noise == "uniform_noise":
            return np.random.uniform(-1.0, 1.0, size=[self.batch_size, self.latent_dim])



    def decode_labels(self, encoded_labels):
        decoded_labels = []
        for labels in encoded_labels:
            for c in range(0, len(self.real_labels)):
                if labels[c]==1:
                    decoded_labels.append(c)
        return decoded_labels

    def train(self, epochs, batch_size=64, sample_interval=50):

        datagen = ImageDataGenerator(
                    # rescale=1./255,
                    rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    horizontal_flip=True)

        real = np.ones((self.batch_size, 1))
        fake = np.zeros((self.batch_size, 1))

        print('started_learning')
        for epoch in range(epochs):
            batches = 0
            for imgs, labels in datagen.flow_from_directory(
                        'C:/Users/trist/Documents/GitHub/DeepLearningProject/dataset-resized',
                        target_size=(self.img_cols, self.img_rows),
                        batch_size=self.batch_size,
                        class_mode='categorical',
                        shuffle=True): #save_to_dir='C:/Users/trist/Downloads/TrashNet/augmented'):
                batches += 1
                labels = np.array(self.decode_labels(labels))
                # convert from ints to floats
                imgs = imgs.astype('float32')
                # scale from [0,255] to [-1,1]
                imgs = (imgs - 127.5) / 127.5
                noise = self.generate_noise("normal_noise", self.batch_size)
                # Generate a half batch of new images
                # we can use labels instead of fake_labels; because it is fake for noise
                try:
                    gen_imgs = self.generator.predict([noise, labels])
                except ValueError:
                    # print('uhm')
                    continue    

                # --------------------- Train the Discriminator ---------------------
                d_loss_real = self.discriminator.train_on_batch([imgs, labels], real)
                d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                #  --------------------- Train the Generator ---------------------
                # Condition on labels (random one-hot labels)
                # fake_labels = np.eye(self.num_classes)[np.random.choice(self.num_classes, self.batch_size)]
                fake_labels = randint(0, self.num_classes, self.batch_size)        
                cgan_loss, acc = self.cgan_model.train_on_batch([noise, fake_labels], real)

                # Plot the progress
                print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], cgan_loss))

                if batches >= 2527 / self.batch_size:
                    # we need to break the loop by hand because
                    # the generator loops indefinitely
                    break

            self.sample_images(epoch)


    def sample_images(self, epoch):
        r, c = 2, 3
        noise = self.generate_noise('normal_noise', 6)
        sampled_labels = np.arange(0, 6).reshape(-1, 1)
        sampled_labels_categorical = to_categorical(sampled_labels)
        decoded_labels = np.array(self.decode_labels(sampled_labels_categorical))
        gen_imgs = self.generator.predict([noise, decoded_labels])


        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(array_to_img(gen_imgs[cnt], scale=True, data_format='channels_last'))
                axs[i, j].set_title("%s" % self.real_labels[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("C:/Users/trist/Downloads/TrashNet/Images/%d.png" % epoch, bbox_inches='tight', dpi=200)
        plt.close()

if __name__ == '__main__':
    cgan = CGAN()
    cgan.train(epochs=10000 , batch_size=64, sample_interval=50)