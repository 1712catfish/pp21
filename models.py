from settings import *

from keras import *
from keras.layers import *


def primeColumns():
    inputs = Input((IMSIZE, IMSIZE, 3))
    x = BACKBONE(inputs)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model
