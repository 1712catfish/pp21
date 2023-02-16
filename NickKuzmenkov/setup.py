from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow as tf
import pandas as pd
import numpy as np
import os

print('Using tensorflow %s' % tf.__version__)

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print('Running on TPUv3-8')
except:
    tpu = None
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    strategy = tf.distribute.get_strategy()
    print('Running on GPU with mixed precision')

batch_size = 16 * strategy.num_replicas_in_sync

print('Number of replicas:', strategy.num_replicas_in_sync)
print('Batch size: %.i' % batch_size)


class BaseSettings:
    strategy = strategy
    batch_size = batch_size

    tf_record_img_size = 600
    classes = ['complex',
               'frog_eye_leaf_spot',
               'powdery_mildew',
               'rust',
               'scab']

    gcs_path_raw = KaggleDatasets().get_gcs_path('pp2021-kfold-tfrecords-0')
    gcs_path_aug = [
        KaggleDatasets().get_gcs_path('pp2021-kfold-tfrecords'),
        KaggleDatasets().get_gcs_path('pp2021-kfold-tfrecords-1'),
        KaggleDatasets().get_gcs_path('pp2021-kfold-tfrecords-2'),
        KaggleDatasets().get_gcs_path('pp2021-kfold-tfrecords-3')
    ]

    seed = 2021
    epochs = 100  # maximum number of epochs <-- keep this large as we use EarlyStopping
    patience = [5, 2]  # patience[0] is for EarlyStopping, patience[1] is for ReduceLROnPlateau
    factor = .1  # new_lr =  lr * factor if patience_count > patience[1]
    min_lr = 1e-8  # minimum optimizer lr

    verbose = 2  # set this to 1 to see live progress bar or to 2 when commiting

    folds = 5  # number of KFold folds
    used_folds = [0, 1, 2, 3, 4]  # number of used folds <-- here we use only the first one
    num_classes = len(classes)


