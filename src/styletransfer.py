# -*- coding: utf-8 -*-

import pickle
import datetime
import os
import math
import glob
import time

import numpy as np

from tensorflow.keras.layers import (
    Conv2D,
    Conv2DTranspose,
    BatchNormalization,
    Add,
    Activation,
    Input,
    Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import img_to_array, array_to_img
from tensorflow.keras.optimizers import Adadelta

from utils import config_gpu, load_img, load_imgs
_ = config_gpu()


#############################################################################
# set params and paths to save models
#############################################################################
input_shape = (224, 224, 3)
batch_size = 2
epochs = 10

dt = datetime.datetime.now()
run_time = str(dt.date()).replace('-', '') + str(dt.hour) + str(dt.minute)

dir_log = '../model/style_transfer/log_{}'.format(run_time)
dir_weights = '../model/style_transfer/weights_{}'.format(run_time)
dir_gen_weights = '../model/style_transfer/gen_weights_{}'.format(run_time)
dir_trans = '../model/style_transfer/trans_{}'.format(run_time)

os.makedirs(dir_log, exist_ok=True)
os.makedirs(dir_weights, exist_ok=True)
os.makedirs(dir_gen_weights, exist_ok=True)
os.makedirs(dir_trans, exist_ok=True)
#############################################################################


def residual_block(input_ts):
    x = Conv2D(
        128,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same'
    )(input_ts)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(
        128,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='same'
    )(x)
    x = BatchNormalization()(x)

    return Add()([x, input_ts])


def build_encoder_decoder(input_shape=(224, 224, 3)):

    # encoder
    input_ts = Input(shape=input_shape, name='input')
    x = Lambda(lambda x: x/255.)(input_ts)

    x = Conv2D(32, (9, 9), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # add 5 residual block
    for _ in range(5):
        x = residual_block(x)

    # decoder
    x = Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(32, (3, 3), strides=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2DTranspose(3, (9, 9), strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)

    # scale pixel value to [0, 255] (tanh returns (-1, 1))
    gen_out = Lambda(lambda x: (x+1)*127.5)(x)

    model_gen = Model(
        inputs=[input_ts],
        outputs=[gen_out]
    )

    return model_gen


# preprocess input features using vgg16
# 1. RGB -> BGR
# 2. centering
def norm_vgg16(x):
    return (x[:, :, :, ::-1] - 120) / 255.


def train_generator(img_paths,
                    batch_size,
                    model,
                    y_true_sty,
                    shuffle=True,
                    epochs=None):

    n_samples = len(img_paths)
    indices = list(range(n_samples))
    steps_per_epoch = math.ceil(n_samples / batch_size)
    img_paths = np.array(img_paths)
    cnt_epoch = 0
    while True:
        cnt_epoch += 1
        if shuffle:
            np.random.shuffle(indices)
        for i in range(steps_per_epoch):
            start = batch_size*i
            end = batch_size*(i+1)
            X = load_imgs(img_paths[indices[start:end]], input_size)
            batch_size_act = X.shape[0]
            y_true_sty_t = [
                np.repeat(feat, batch_size_act, axis=0)
                for feat in y_true_sty
            ]
            # exrtract content
            y_true_con = model.predict(X)
            yield (X, y_true_sty_t + [y_true_con])

        if epochs is not None:
            if cnt_epoch >= epochs:
                raise StopIteration


# content loss
def feature_loss(y_true, y_pred):
    norm = K.prod(K.cast(K.shape(y_true)[1:], 'float32'))
    return K.sum(K.square(y_pred - y_true), axis=(1, 2, 3)) / norm


# gram matrix
def gram_matrix(X):

    # like np.transform
    X_sw = K.permute_dimensions(
        X, (0, 3, 2, 1)
    )
    # like np.reshape
    s = K.shape(X_sw)
    new_shape = (s[0], s[1], s[2]*s[3])
    X_rs = K.reshape(X_sw, new_shape)
    X_rs_t = K.permute_dimensions(
        X_rs, (0, 2, 1)
    )
    dot = K.batch_dot(X_rs, X_rs_t)
    norm = K.prod(K.cast(s[1:], 'float32'))

    return dot/norm


# style loss
def style_loss(y_true, y_pred):
    return K.sum(K.square(gram_matrix(y_pred) - gram_matrix(y_true)),
                 axis=(1, 2))


# load pre-trained VGG16
vgg16 = VGG16()

# does not allow re-training the weights
for layer in vgg16.layers:
    layer.trainable = False


# names for layers extracting features
style_layer_names = (
    'block1_conv2',
    'block2_conv2',
    'block3_conv3',
    'block4_conv3'
)
content_layer_names = (
    'block3_conv3',
)
# preserves outputs of middle layer
style_outputs_gen = []
content_outputs_gen = []


# pass output image of style-transfer network as a input image
model_gen = build_encoder_decoder(input_shape)
input_gen = model_gen.output


# normalize input image
z = Lambda(norm_vgg16)(input_gen)
for layer in vgg16.layers:
    z = layer(z)
    if layer.name in style_layer_names:
        style_outputs_gen.append(z)
    if layer.name in content_layer_names:
        content_outputs_gen.append(z)


model = Model(
    inputs=model_gen.input,
    outputs=style_outputs_gen + content_outputs_gen
)

input_size = input_shape[:2]
img_sty = load_img(
    '../data/chap11/img/style/Piet_Mondrian_Composition.png',
    target_size=input_size
)
img_arr_sty = np.expand_dims(img_to_array(img_sty), axis=0)


# extract style features from true image
input_sty = Input(shape=input_shape, name='input_sty')
style_outputs = []
x = Lambda(norm_vgg16)(input_sty)

for layer in vgg16.layers:
    x = layer(x)
    if layer.name in style_layer_names:
        style_outputs.append(x)

model_sty = Model(
    inputs=input_sty,
    outputs=style_outputs
)
y_true_sty = model_sty.predict(img_arr_sty)


# extract content features from true image
input_con = Input(shape=input_shape, name='input_con')
content_outputs = []
y = Lambda(norm_vgg16)(input_con)

for layer in vgg16.layers:
    y = layer(y)
    if layer.name in content_layer_names:
        content_outputs.append(y)

model_con = Model(
    inputs=input_con,
    outputs=content_outputs
)
# y_true_con = model_con.predict(img_arr_sty)


# training images
path_glob = os.path.join('../data/chap11/img/context/*.jpg')
img_paths = glob.glob(path_glob)

gen = train_generator(
    img_paths,
    batch_size,
    model_con,
    y_true_sty,
    epochs=epochs
)


model.compile(
    optimizer=Adadelta(),
    loss=[
        style_loss,
        style_loss,
        style_loss,
        style_loss,
        feature_loss
    ],
    loss_weights=[1.0, 1.0, 1.0, 1.0, 3.0]
)

# save immediate style-transfered result during training time
img_test = load_img(
    '../data/chap11/img/test/building.jpg',
    target_size=input_size
)
img_arr_test = np.expand_dims(img_to_array(img_test), axis=0)

# minibatch steps
steps_per_epoch = math.ceil(len(img_paths)/batch_size)

iters_verbose = 1000
iters_save_img = iters_verbose
iters_save_model = steps_per_epoch


# run model
now_epoch = 0
losses = []
path_tmp = 'epoch_{}_iters_{}_losses_{:.2f}_{}'


start_time = time.time()
for i, (x_train, y_train) in enumerate(gen):

    if i % steps_per_epoch == 0:
        now_epoch += 1

    # training
    loss = model.train_on_batch(x_train, y_train)
    losses.append(loss)

    # print training progressing result
    if i % iters_verbose == 0:
        print('epoch: {}, iters: {}, loss: {:.3f}'.
              format(now_epoch, i, loss[0]))

    # save image
    if i % iters_save_img == 0:
        pred = model_gen.predict(img_arr_test)
        img_pred = array_to_img(pred.squeeze())
        path_trs_img = path_tmp.format(
            now_epoch, i, loss[0], '.jpg'
        )
        img_pred.save(
            os.path.join(dir_trans, path_trs_img)
        )
        print('# image saved: {}'.format(path_trs_img))
        print('time elapsed: {} sec\n'.format(time.time() - start_time))

    # save model, losses
    if i % iters_save_model == 0:
        model.save(
            os.path.join(
                dir_weights,
                path_tmp.format(now_epoch, i, loss[0], '.h5')
            )
        )
        model_gen.save(
            os.path.join(
                dir_gen_weights,
                path_tmp.format(now_epoch, i, loss[0], '.h5')
            )
        )
        path_loss = os.path.join(dir_log, 'loss.pkl')
        with open(path_loss, 'wb') as f:
            pickle.dump(losses, f)
