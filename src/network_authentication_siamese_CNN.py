import numpy as np
import util
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K

np.random.seed(813306)

def set_layers(input, lead_i):

    left = layers.Cropping2D(cropping=((0, 0), (0, 1)))(input)
    right = layers.Cropping2D(cropping=((0, 0), (1, 0)))(input)

    model = models.Sequential()
    model.add(layers.Reshape((25, 12 if not lead_i else 1, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(256, (7, 1), strides=(4, 1), padding="same", kernel_initializer="he_normal"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Conv2D(512, (5, 1), strides=(2, 1), padding="same", kernel_initializer="he_normal"))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))

    left_ = model(left)
    right_ = model(right)

    left_right = layers.Lambda(lambda x_lr: K.abs(x_lr[0] - x_lr[1]))
    lr_distance = left_right([left_, right_])

    x = layers.Dropout(0.9)(lr_distance)
    x = layers.Dense(16)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.Dense(2, activation='softmax')(x)
    return x


def build_network(**params):
    lead_i =params.get('lead_i', False)
    i = layers.Input(shape=[600, 2, 1] if not lead_i else [50, 2, 1])
    o = set_layers(i, lead_i)
    print('        -- model was built.')
    model = models.Model(inputs=i, outputs=o)

    model = util.add_compile(model, **params)
    return model
