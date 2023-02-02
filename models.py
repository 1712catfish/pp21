from keras import *
from keras.layers import *
from keras.applications import *

from settings import *


def create_plant_pathology_model():
    inputs = InputLayer((IMSIZE, IMSIZE, 3))

    x = MobileNetV3Small(
        weights='imagenet',
        minimalistic=True,
        include_top=False,
        droput_rate=0.2,
    )(inputs)
    x = MultiHeadAttention(num_heads=32, key_dim=2)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)

    return model
