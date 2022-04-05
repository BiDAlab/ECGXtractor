import numpy as np
from tensorflow.keras import layers, models
import util

np.random.seed(813306)


def encoder(input):
    x = layers.Conv2D(8, (5, 1), strides=(1, 1), padding='same',
                                 kernel_initializer="he_normal")(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(16, (3, 1), strides=(1, 1), padding='same',
                            kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='same')(x)

    x = layers.Conv2D(32, (3, 1), strides=(1, 1), padding='same',
                            kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(64, (3, 1), strides=(1, 1), padding='same',
                            kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='same')(x)

    x = layers.Conv2D(128, (3, 1), strides=(1, 1), padding='same',
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(256, (3, 1), strides=(1, 1), padding='same',
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='same')(x)

    x = layers.Conv2D(512, (3, 1), strides=(1, 1), padding='same',
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(1024, (3, 1), strides=(1, 1), padding='same',
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 1), strides=(2, 1), padding='same')(x)

    x = layers.Conv2D(2, (3, 1), strides=(1, 1), padding='same',
                      kernel_initializer="he_normal")(x)
    x = layers.Flatten()(x)

    return x


def decoder(input):
    x = layers.Reshape((25, 12, 2))(input)
    x = layers.Conv2D(2, (3, 1), strides=(1, 1), padding='same',
                      kernel_initializer="he_normal")(x)

    x = layers.Conv2D(1024, (3, 1), strides=(1, 1), padding='same',
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.UpSampling2D(size=(2, 1))(x)

    x = layers.Conv2D(512, (3, 1), strides=(1, 1), padding='same',
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(256, (3, 1), strides=(1, 1), padding='same',
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.UpSampling2D(size=(2, 1))(x)

    x = layers.Conv2D(128, (3, 1), strides=(1, 1), padding='same',
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(64, (3, 1), strides=(1, 1), padding='same',
                            kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.UpSampling2D(size=(2, 1))(x)

    x = layers.Conv2D(32, (3, 1), strides=(1, 1), padding='same',
                            kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(16, (3, 1), strides=(1, 1), padding='same',
                            kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.UpSampling2D(size=(2, 1))(x)

    x = layers.Conv2D(8, (5, 1), strides=(1, 1), padding='same',
                            kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(1, (3, 1), strides=(1, 1), padding='same',
                            kernel_initializer="he_normal")(x)
    x = layers.Flatten()(x)

    x = layers.Reshape((400, 12, 1))(x)
    return x


def build_network(**params):
    i = layers.Input(shape=[400, 12, 1])
    e = encoder(i)
    o = decoder(e)
    print('        -- model was built.')
    model = models.Model(inputs=i, outputs=o)

    model = util.add_compile(model, **params)
    return model

