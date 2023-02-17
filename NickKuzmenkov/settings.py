# import sys
# sys.path.append('/kaggle/input/efficientnet-keras-dataset/efficientnet_kaggle')
from keras_cv_attention_models import *
# from kaggle_datasets import KaggleDatasets
from sklearn.model_selection import KFold
# import efficientnet.tfkeras as efn
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from keras_cv_attention_models import *
import tensorflow as tf
import pandas as pd
# from keras_cv.layers import RandAugment
import os
import numpy as np
from keras_cv_attention_models import *
import tensorflow_addons as tfa
from sklearn.metrics import *
# from imgaug import augmenters as iaa
# import imgaug as ia
import random, re, math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import tensorflow as tf
print('Tensorflow version ' + tf.__version__)
from sklearn.model_selection import KFold
from tensorflow.keras.layers import *

tf.random.set_seed(42)
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


class Settings:
    strategy = strategy
    batch_size = batch_size

    tf_record_img_size = 600
    classes = ['complex',
               'frog_eye_leaf_spot',
               'powdery_mildew',
               'rust',
               'scab']

    # print("Getting gcs paths...")
    # gcs_path_raw = KaggleDatasets().get_gcs_path('pp2021-kfold-tfrecords-0')
    gcs_path_raw = 'gs://kds-05fece17f39b6a102cff4f693d3cdb66fcc3817fb1400871dbbaacb6'
    gcs_path_aug = [
        # KaggleDatasets().get_gcs_path('pp2021-kfold-tfrecords'),
        # KaggleDatasets().get_gcs_path('pp2021-kfold-tfrecords-1'),
        # KaggleDatasets().get_gcs_path('pp2021-kfold-tfrecords-2'),
        # KaggleDatasets().get_gcs_path('pp2021-kfold-tfrecords-3'),

        'gs://kds-b9244f883ccd546bdb04081a759e0c8f1747c9aaf21d6c80a04d3a83',
        'gs://kds-651f9f5bbd8fba5ada0b93f9ca147f64c53155860bffc87fc282c647',
        'gs://kds-c3fb887549239fc51a6a5b9aa27dacc9edacb2cd2e301e30688c9891',
        'gs://kds-a4ac385afbcd046dd66e6910b3a764acbd72867cae4c154a60304ab1'
    ]

    seed = 42
    epochs = 100  # maximum number of epochs <-- keep this large as we use EarlyStopping
    patience = [5, 2]  # patience[0] is for EarlyStopping, patience[1] is for ReduceLROnPlateau
    factor = .1  # new_lr =  lr * factor if patience_count > patience[1]
    min_lr = 1e-8  # minimum optimizer lr

    verbose = 1  # set this to 1 to see live progress bar or to 2 when commiting

    folds = 5  # number of KFold folds
    used_folds = [0, 1, 2, 3, 4]  # number of used folds <-- here we use only the first one
    num_classes = len(classes)




