# -*- coding: utf-8 -*-

import glob
import os
import math
import logging

import cv2 as cv
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.callbacks import ModelCheckpoint

from utils import config_gpu, img_to_array, load_img
_ = config_gpu()


img_size = 224
batch_size = 128
epochs = 100

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def rgb2lab(rgb):
    assert rgb.dtype == 'uint8'

    return cv.cvtColor(rgb, cv.COLOR_RGB2LAB)


def lab2rgb(lab):
    assert lab.dtype == 'uint8'

    return cv.cvtColor(lab, cv.COLOR_LAB2RGB)


def get_lab_from_data_list(data_list):

    x_lab = []
    for f in data_list:
        rgb = img_to_array(
            load_img(f, target_size=(img_size, img_size))
        ).astype(np.uint8)
        lab = rgb2lab(rgb)
        x_lab.append(lab)

    return np.stack(x_lab)


def generator_with_preprocessing(data_list, batch_size, shuffle=False):

    while True:
        if shuffle:
            np.random.shuffle(data_list)
        for i in range(0, len(data_list), batch_size):
            batch_list = data_list[i:i+batch_size]
            batch_lab = get_lab_from_data_list(batch_list)
            batch_l = batch_lab[:, :, :, 0:1]
            batch_ab = batch_lab[:, :, :, 1:]
            yield (batch_l, batch_ab)


data_path = '../data/chap9/img/colorize/'
data_lists = glob.glob(os.path.join(data_path, '*.jpg'))

val_n_sample = math.floor(len(data_lists)*0.1)
test_n_sample = math.floor(len(data_lists)*0.1)
train_n_sample = len(data_lists) - val_n_sample - test_n_sample

val_lists = data_lists[:val_n_sample]
test_lists = data_lists[val_n_sample:(val_n_sample+test_n_sample)]
train_lists = data_lists[(val_n_sample+test_n_sample):
                         (train_n_sample+val_n_sample+test_n_sample)]


train_gen = generator_with_preprocessing(train_lists, batch_size, shuffle=True)
val_gen = generator_with_preprocessing(val_lists, batch_size, shuffle=False)
test_gen = generator_with_preprocessing(test_lists, batch_size, shuffle=False)

train_steps = math.ceil(len(train_lists)/batch_size)
val_steps = math.ceil(len(val_lists)/batch_size)
test_steps = math.ceil(len(test_lists)/batch_size)


autoencoder = Sequential()
# encoder
# (224, 224, 1) -> (224, 224, 32)
autoencoder.add(
    Conv2D(
        32,
        (3, 3),
        (1, 1),
        activation='relu',
        padding='same',
        input_shape=(224, 224, 1)
    )
)
# (224, 224, 32) -> (112, 112, 64)
autoencoder.add(
    Conv2D(
        64,
        (3, 3),
        (2, 2),
        activation='relu',
        padding='same'
    )
)
# (112, 112, 64) -> (56, 56, 128)
autoencoder.add(
    Conv2D(
        128,
        (3, 3),
        (2, 2),
        activation='relu',
        padding='same'
    )
)
# (56, 56, 128) -> (28, 28, 256)
autoencoder.add(
    Conv2D(
        256,
        (3, 3),
        (2, 2),
        activation='relu',
        padding='same'
    )
)
# decoder
# (28, 28, 256) -> (56, 56, 128)
autoencoder.add(
    Conv2DTranspose(
        128,
        (3, 3),
        (2, 2),
        activation='relu',
        padding='same'
    )
)
# (56, 56, 128) -> (112, 112, 64)
autoencoder.add(
    Conv2DTranspose(
        64,
        (3, 3),
        (2, 2),
        activation='relu',
        padding='same'
    )
)
# (112, 112, 64) -> (224, 224, 32)
autoencoder.add(
    Conv2DTranspose(
        32,
        (3, 3),
        (2, 2),
        activation='relu',
        padding='same'
    )
)
# (224, 224, 32) -> (224, 224, 2)
autoencoder.add(
    Conv2D(
        2,
        (1, 1),
        (1, 1),
        activation='relu',
        padding='same'
    )
)
autoencoder.compile(optimizer='adam', loss='mse')

cp_path = '../model/autoencoder_colorize/'
cp_callback = ModelCheckpoint(cp_path, save_weights_only=True, verbose=0)

autoencoder.fit_generator(
    generator=train_gen,
    steps_per_epoch=train_steps,
    epochs=epochs,
    validation_data=val_gen,
    validation_steps=val_steps,
    callbacks=[cp_callback]
)
