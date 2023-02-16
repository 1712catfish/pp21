from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow as tf
import pandas as pd
import numpy as np
import os

class ExtraSettings:
    model_img_size = 280

    @staticmethod
    def get_model():
        model = tf.keras.models.Sequential(name='EfficientNetB4')

        model.add(efn.EfficientNetB4(
            include_top=False,
            input_shape=(ExtraSettings.model_img_size, ExtraSettings.model_img_size, 3),
            weights='noisy-student',
            pooling='avg'))

        model.add(tf.keras.layers.Dense(
            BaseSettings.num_classes,
            kernel_initializer=tf.keras.initializers.RandomUniform(seed=BaseSettings.seed),
            bias_initializer=tf.keras.initializers.Zeros(), name='dense_top')
        )
        model.add(tf.keras.layers.Activation('sigmoid', dtype='float32'))

        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(),
            optimizer='adam',
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name='acc'),
                tfa.metrics.F1Score(
                    num_classes=len(BaseSettings.classes),
                    average='macro')
            ]
        )

        return model