# -*- coding: utf-8 -*-

import os
import time

import numpy as np

from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator, array_to_img
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, Flatten, Dense, Reshape, UpSampling2D, Input
)
from tensorflow.keras.losses import mean_absolute_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution

from utils import config_gpu
_ = config_gpu()
disable_eager_execution()


#############################################
DATA_DIR = '../data/chap12/data/chap12/'
BATCH_SIZE = 16
IMG_SHAPE = (64, 64, 3)


n_filters = 64
n_layers = 4
z_size = 32


GAMMA = 0.5
LR_K = 0.001
TOTAL_STEPS = 100000


MODEL_SAVE_DIR = '../model/began/models'
IMG_SAVE_DIR = '../model/began/imgs'
LOG_DIR = '../model/began/logs'

# generate 5x5 image for verification
IMG_SAMPLE_SHAPE = (5, 5)
N_IMG_SAMPLES = np.prod(IMG_SAMPLE_SHAPE)

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(IMG_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


sample_seeds = np.random.uniform(
    -1, 1, (N_IMG_SAMPLES, z_size)
)
history = []
logs = []

#############################################


def build_encoder(input_shape, z_size, n_filters, n_layers):
    """build_encoder

    Parameters
    ----------

    input_shape : input image
    z_size : dim feature vectors
    n_filters : number of files
    n_layers : number of layers

    Returns
    -------
    """

    model = Sequential()
    model.add(
        Conv2D(
            n_filters,
            3,
            activation='elu',
            input_shape=input_shape,
            padding='same'
        )
    )
    model.add(Conv2D(n_filters, 3, padding='same'))
    for i in range(2, n_layers+1):
        model.add(
            Conv2D(
                i*n_filters,
                3,
                activation='elu',
                padding='same'
            )
        )
        model.add(
            Conv2D(
                i*n_filters,
                3,
                strides=2,
                activation='elu',
                padding='same'
            )
        )
    model.add(Conv2D(n_layers*n_filters, 3, padding='same'))
    model.add(Flatten())
    model.add(Dense(z_size))

    return model


def build_decoder(output_shape, z_size, n_filters, n_layers):
    """build_decoder

    Parameters
    ----------

    output_shape :
    z_size :
    n_filters :
    n_layers :

    Returns
    -------
    """

    # used in upsampling
    scale = 2**(n_layers-1)

    # calculate input size
    fc_shape = (
        output_shape[0] // scale,
        output_shape[1] // scale,
        n_filters
    )
    fc_size = fc_shape[0]*fc_shape[1]*fc_shape[2]

    model = Sequential()
    # fc layers
    model.add(Dense(fc_size, input_shape=(z_size, )))
    model.add(Reshape(fc_shape))

    for i in range(n_layers-1):
        model.add(
            Conv2D(
                n_filters,
                3,
                activation='elu',
                padding='same'
            )
        )
        model.add(
            Conv2D(
                n_filters,
                3,
                activation='elu',
                padding='same'
            )
        )
        model.add(UpSampling2D())

    # last layer does not need upsampling
    model.add(
        Conv2D(
            n_filters,
            3,
            activation='elu',
            padding='same'
        )
    )
    model.add(
        Conv2D(
            n_filters,
            3,
            activation='elu',
            padding='same'
        )
    )
    # outputs 3-channel image
    model.add(Conv2D(3, 3, padding='same'))

    return model


def build_generator(img_shape, z_size, n_filters, n_layers):

    decoder = build_decoder(
        img_shape, z_size, n_filters, n_layers
    )

    return decoder


def build_discriminator(img_shape, z_size, n_filters, n_layers):

    encoder = build_encoder(
        img_shape, z_size, n_filters, n_layers
    )
    decoder = build_decoder(
        img_shape, z_size, n_filters, n_layers
    )

    return Sequential((encoder, decoder))


def build_discriminator_trainer(discriminator):

    img_shape = discriminator.input_shape[1:]
    real_inputs = Input(img_shape)
    fake_inputs = Input(img_shape)
    real_outputs = discriminator(real_inputs)
    fake_outputs = discriminator(fake_inputs)

    return Model(inputs=[real_inputs, fake_inputs],
                 outputs=[real_outputs, fake_outputs])


def build_generator_loss(discriminator):

    def loss(y_true, y_pred):
        reconst = discriminator(y_pred)
        return mean_absolute_error(
            reconst,
            y_pred
        )

    return loss


# convergence test
def measure(real_loss, fake_loss, gamma):

    return real_loss + np.abs(gamma*real_loss - fake_loss)


def save_imgs(path, imgs, rows, cols):

    base_width = imgs.shape[1]
    base_height = imgs.shape[2]
    channels = imgs.shape[3]
    output_shape = (
        base_height*rows,
        base_width*cols,
        channels
    )
    buffer = np.zeros(output_shape)
    for row in range(rows):
        for col in range(cols):
            img = imgs[row*cols + col]
            buffer[
                row*base_height:(row + 1)*base_height,
                col*base_width:(col + 1)*base_width
            ] = img
    array_to_img(buffer).save(path)


data_gen = ImageDataGenerator(rescale=1/255.)

train_data_generator = data_gen.flow_from_directory(
    directory=DATA_DIR,
    classes=['faces'],
    class_mode=None,
    batch_size=BATCH_SIZE,
    target_size=IMG_SHAPE[:2]
)


# build model
generator = build_generator(
    IMG_SHAPE, z_size, n_filters, n_layers
)
discriminator = build_discriminator(
    IMG_SHAPE, z_size, n_filters, n_layers
)
discriminator_trainer = build_discriminator_trainer(discriminator)

# compile generator
# initial learning rate
g_lr = 0.0001

generator_loss = build_generator_loss(discriminator)
generator.compile(
    loss=generator_loss,
    optimizer=Adam(g_lr)
)

# compile discriminator
# initial learning rate
d_lr = 0.0001

k_var = 0.0
k = K.variable(k_var)
discriminator_trainer.compile(
    loss=[
        mean_absolute_error,
        mean_absolute_error
    ],
    loss_weights=[1., -k],
    optimizer=Adam(d_lr)
)


# train models
start_time = time.time()
for step, batch in enumerate(train_data_generator):

    # if the number of images are not multiple of batch_size
    if len(batch) < BATCH_SIZE:
        continue

    # quit traning
    if step > TOTAL_STEPS:
        break

    z_g = np.random.uniform(
        -1, 1, (BATCH_SIZE, z_size)
    )
    z_d = np.random.uniform(
        -1, 1, (BATCH_SIZE, z_size)
    )

    # generate image to use for training generator
    g_pred = generator.predict(z_d)

    # train generator 1-step
    generator.train_on_batch(z_g, batch)

    # train discriminator 1-step
    _, real_loss, fake_loss = discriminator_trainer.train_on_batch(
        [batch, g_pred],
        [batch, g_pred]
    )

    # update k
    k_var += LR_K*(GAMMA*real_loss - fake_loss)
    K.set_value(k, k_var)

    # save loss for calculating g_measure
    history.append({
        'real_loss': real_loss,
        'fake_loss': fake_loss,
    })

    # print log for each 1000 iterations
    if step % 1000 == 0:
        # loss for past 1000 measures
        measurement = np.mean([
            measure(loss['real_loss'],
                    loss['fake_loss'],
                    GAMMA)
            for loss in history[-1000:]
        ])

        # print current log
        logs.append({
            'k': K.get_value(k),
            'measure': measurement,
            'real_loss': real_loss,
            'fake_loss': fake_loss
        })
        print(logs[-1])

        # save image
        img_path = '{}/generated_{}.png'.format(
            IMG_SAVE_DIR,
            step
        )
        save_imgs(
            img_path,
            generator.predict(sample_seeds),
            rows=IMG_SAMPLE_SHAPE[0],
            cols=IMG_SAMPLE_SHAPE[1]
        )

        # save recent model
        generator.save('{}/generator_{}.hd5'.
                       format(MODEL_SAVE_DIR, step))
        discriminator.save('{}/discriminator_{}.hd5'.
                           format(MODEL_SAVE_DIR, step))

        print('time elapsed: {} sec ({}/{})\n'.
              format(time.time() - start_time,
                     step, TOTAL_STEPS))
