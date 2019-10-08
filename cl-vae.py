from __future__ import print_function
import collections
import os
import tensorflow as tf
import keras
from keras import Model
from keras import backend as K
from keras.models import Sequential, load_model
from keras.datasets import mnist
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed, Input, concatenate, LeakyReLU, Reshape
from keras.layers import LSTM, Bidirectional, GRU, Conv1D
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, Flatten, Lambda, Layer
from keras.optimizers import Adam, RMSprop
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.backend import clear_session
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd

batch_size = 100
latent_dim = 2
epochs = 100
img_dim = 28
filters = 16
intermediate_dim = 256

img_dim = 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, img_dim, img_dim, 1))
x_test = x_test.reshape((-1, img_dim, img_dim, 1))
num_classes = len(np.unique(y_train))

x = Input(shape=(img_dim, img_dim, 1))
h = x

for i in range(2):
    filters *= 2
    h = Conv2D(filters=filters,
            kernel_size=3,
            strides=2,
            padding='same')(h)
    h = LeakyReLU(0.2)(h)
    h = Conv2D(filters=filters,
            kernel_size=3,
            strides=1,
            padding='same')(h)
    h = LeakyReLU(0.2)(h)


h_shape = K.int_shape(h)[1:]
h = Flatten()(h)
z_mean = Dense(latent_dim)(h) 
z_log_var = Dense(latent_dim)(h) 

clvae_encoder = Model(x, z_mean) 


z = Input(shape=(latent_dim,))
h = z
h = Dense(np.prod(h_shape))(h)
h = Reshape(h_shape)(h)

for i in range(2):
    h = Conv2DTranspose(filters=filters,
                        kernel_size=3,
                        strides=1,
                        padding='same')(h)
    h = LeakyReLU(0.2)(h)
    h = Conv2DTranspose(filters=filters,
                        kernel_size=3,
                        strides=2,
                        padding='same')(h)
    h = LeakyReLU(0.2)(h)
    filters //= 2

x_recon = Conv2DTranspose(filters=1,
                        kernel_size=3,
                        activation='sigmoid',
                        padding='same')(h)

clvae_decoder = Model(z, x_recon) 

y_in = Input(shape=(num_classes,), name='input_y')

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,), name='sampling')([z_mean, z_log_var])
x_recon = clvae_decoder(z)

class Gaussian(Layer):
    """A simple layer that computes the stats for each Gaussian outputs the mean.
    """
    def __init__(self, num_classes, **kwargs):
        self.num_classes = num_classes
        super(Gaussian, self).__init__(**kwargs)
        
    def build(self, input_shape):
        latent_dim = input_shape[-1]
        self.mean = self.add_weight(name='mean',
                                    shape=(self.num_classes, latent_dim),
                                    initializer='zeros')
        
    def call(self, inputs):
        z = inputs # z.shape=(batch_size, latent_dim)
        z = K.expand_dims(z, 1)
        return z - K.expand_dims(self.mean, 0)
    
    def compute_output_shape(self, input_shape):
        return (None, self.num_classes, input_shape[-1])

gaussian = Gaussian(num_classes, name='priors')
z_prior_mean = gaussian(z)

clvae = Model([x, y_in], [x_recon, z_prior_mean])

z_mean = K.expand_dims(z_mean, 1)
z_log_var = K.expand_dims(z_log_var, 1)

lamb = 0.5
xent_loss = 0.5 * K.mean((x - x_recon)**2, 0)
kl_loss = - 0.5 * (z_log_var - K.square(z_prior_mean))
kl_loss = K.mean(K.batch_dot(K.expand_dims(y_in, 1), kl_loss), 0)
clvae_loss = lamb * K.sum(xent_loss) + K.sum(kl_loss)

clvae.add_loss(clvae_loss)
clvae.compile(optimizer='adam')
clvae.summary()

clvae_history = clvae.fit([x_train, to_categorical(y_train)],
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([x_test, to_categorical(y_test)], None))