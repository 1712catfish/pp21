from kaggle_datasets import KaggleDatasets
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


class CFG():
    '''
    keep these
    '''
    strategy = strategy
    batch_size = batch_size

    img_size = 600
    classes = [
        'complex',
        'frog_eye_leaf_spot',
        'powdery_mildew',
        'rust',
        'scab']

    gcs_path_raw = KaggleDatasets().get_gcs_path('pp2021-kfold-tfrecords-0')

    gcs_path_aug = [
        KaggleDatasets().get_gcs_path('pp2021-kfold-tfrecords'),
        KaggleDatasets().get_gcs_path('pp2021-kfold-tfrecords-1'),
        KaggleDatasets().get_gcs_path('pp2021-kfold-tfrecords-2'),
        KaggleDatasets().get_gcs_path('pp2021-kfold-tfrecords-3')]

    '''
    tweak these
    '''
    seed = 42  # random seed we use for each operation
    epochs = 100  # maximum number of epochs <-- keep this large as we use EarlyStopping
    patience = [5, 2]  # patience[0] is for EarlyStopping, patience[1] is for ReduceLROnPlateau
    factor = .1  # new_lr =  lr * factor if patience_count > patience[1]
    min_lr = 1e-8  # minimum optimizer lr

    verbose = 2  # set this to 1 to see live progress bar or to 2 when commiting

    folds = 5  # number of KFold folds
    used_folds = [0, 1, 2, 3, 4]  # number of used folds <-- here we use only the first one


def count_data_items(filenames):
    return np.sum([int(x[:-6].split('-')[-1]) for x in filenames])


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.reshape(image, [CFG.img_size, CFG.img_size, 3])
    image = tf.cast(image, tf.float32) / 255.
    return image


feature_map = {
    'image': tf.io.FixedLenFeature([], tf.string),
    'image_name': tf.io.FixedLenFeature([], tf.string),
    'complex': tf.io.FixedLenFeature([], tf.int64),
    'frog_eye_leaf_spot': tf.io.FixedLenFeature([], tf.int64),
    'powdery_mildew': tf.io.FixedLenFeature([], tf.int64),
    'rust': tf.io.FixedLenFeature([], tf.int64),
    'scab': tf.io.FixedLenFeature([], tf.int64),
    'healthy': tf.io.FixedLenFeature([], tf.int64)}


def read_tfrecord(example, labeled=True):
    example = tf.io.parse_single_example(example, feature_map)
    image = decode_image(example['image'])
    if labeled:
        label = [tf.cast(example[x], tf.float32) for x in CFG.classes]
    else:
        label = example['image_name']
    return image, label


def get_dataset(filenames, labeled=True, ordered=True, shuffled=False,
                repeated=False, cached=False, distributed=True):
    auto = tf.data.experimental.AUTOTUNE
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=auto)
    if not ordered:
        ignore_order = tf.data.Options()
        ignore_order.experimental_deterministic = False
        dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(
        lambda x: read_tfrecord(x, labeled=labeled),
        num_parallel_calls=auto)
    if shuffled:
        dataset = dataset.shuffle(2048, seed=CFG.seed)
    if repeated:
        dataset = dataset.repeat()
    dataset = dataset.batch(CFG.batch_size)
    if cached:
        dataset = dataset.cache()
    dataset = dataset.prefetch(auto)
    if distributed:
        dataset = CFG.strategy.experimental_distribute_dataset(dataset)
    return dataset


def get_model():
    model = tf.keras.models.Sequential(name='EfficientNetB4')

    model.add(efn.EfficientNetB4(
        include_top=False,
        input_shape=(CFG.img_size, CFG.img_size, 3),
        weights='noisy-student',
        pooling='avg'))

    model.add(tf.keras.layers.Dense(len(CFG.classes),
                                    kernel_initializer=tf.keras.initializers.RandomUniform(seed=CFG.seed),
                                    bias_initializer=tf.keras.initializers.Zeros(), name='dense_top'))
    model.add(tf.keras.layers.Activation('sigmoid', dtype='float32'))

    return model


# histories = []
# scores = []
# image_names = np.empty((0,))
# predicts = np.empty((0, len(CFG.classes)))
#
# callbacks = [
#     tf.keras.callbacks.EarlyStopping(
#         monitor='val_f1_score', mode='max',
#         patience=CFG.patience[0], restore_best_weights=True),
#     tf.keras.callbacks.ReduceLROnPlateau(
#         monitor='val_f1_score', mode='max',
#         patience=CFG.patience[1], min_lr=CFG.min_lr, verbose=2)
# ]
#
# kfold = KFold(n_splits=CFG.folds, shuffle=True, random_state=CFG.seed)
# folds = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']

'''
run training loop
'''

def fit():

    for i, (train_index, val_index) in enumerate(kfold.split(folds)):

        '''
        run only selected folds
        '''
        if i in CFG.used_folds:

            print('=' * 74)
            print(f'Fold {i}')
            print('=' * 74)

            '''
            reinitialize the system
            '''
            if tpu is not None:
                tf.tpu.experimental.initialize_tpu_system(tpu)

            '''
            model setup
            '''
            with CFG.strategy.scope():
                model = get_model()

                model.compile(
                    loss=tf.keras.losses.BinaryCrossentropy(),
                    optimizer='adam',
                    metrics=[
                        tf.keras.metrics.BinaryAccuracy(name='acc'),
                        tfa.metrics.F1Score(
                            num_classes=len(CFG.classes),
                            average='macro')])

            '''
            data setup
            '''
            train_filenames = []
            for j in train_index:
                train_filenames += tf.io.gfile.glob(os.path.join(CFG.gcs_path_aug[0], folds[j], '*.tfrec'))
                train_filenames += tf.io.gfile.glob(os.path.join(CFG.gcs_path_aug[1], folds[j], '*.tfrec'))
                train_filenames += tf.io.gfile.glob(os.path.join(CFG.gcs_path_aug[2], folds[j], '*.tfrec'))
                train_filenames += tf.io.gfile.glob(os.path.join(CFG.gcs_path_aug[3], folds[j], '*.tfrec'))
            np.random.shuffle(train_filenames)

            val_filenames = []
            for j in val_index:
                val_filenames += tf.io.gfile.glob(os.path.join(CFG.gcs_path_raw, folds[j], '*.tfrec'))

            train_dataset = get_dataset(
                train_filenames,
                ordered=False, shuffled=True, repeated=True)

            val_dataset = get_dataset(
                val_filenames,
                cached=True)

            steps_per_epoch = count_data_items(train_filenames) // (20 * CFG.batch_size)
            validation_steps = count_data_items(val_filenames) // CFG.batch_size

            '''
            fit
            '''
            history = model.fit(
                train_dataset,
                steps_per_epoch=steps_per_epoch,
                validation_data=val_dataset,
                validation_steps=validation_steps,
                callbacks=callbacks,
                epochs=CFG.epochs,
                verbose=CFG.verbose).history

            '''
            write out-of-fold predictions
            '''
            size = count_data_items(val_filenames)
            steps = size // CFG.batch_size + 1

            val_dataset = get_dataset(val_filenames, labeled=False, distributed=False)
            val_predicts = model.predict(
                val_dataset.map(lambda x, y: x),
                steps=steps,
                verbose=CFG.verbose)[:size]
            val_image_names = [x.decode() for x in val_dataset.map(lambda x, y: y).unbatch().take(size).as_numpy_iterator()]

            image_names = np.concatenate((image_names, val_image_names))
            predicts = np.concatenate((predicts, val_predicts))

            '''
            finalize
            '''
            model.save_weights(f'model_{i}.h5')
            histories.append(pd.DataFrame(history))
            scores.append(histories[-1]['val_f1_score'].max())

        else:
            pass


def display_scores(scores):
    scores_df = pd.DataFrame({
        'fold': np.arange(len(scores)),
        'f1': np.round(scores, 4)})

    with pd.option_context('display.max_rows', None):
        display(scores_df)

    print('CV %.4f' % scores_df['f1'].mean())


figure, axes = plt.subplots(1, 5, figsize=[20, 5])


def plot_history():
    for i in range(CFG.folds):

        try:
            axes[i].plot(histories[i].loc[:, 'f1_score'], label='train')
            axes[i].plot(histories[i].loc[:, 'val_f1_score'], label='val')
            axes[i].legend()
        except IndexError:
            pass

        axes[i].set_title(f'fold {i}')
        axes[i].set_xlabel('epochs')

    plt.show()
