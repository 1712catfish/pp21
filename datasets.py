import pandas as pd
import tensorflow as tf
from keras_cv.layers import RandAugment
from sklearn.model_selection import train_test_split

from settings import *


# datagen = ImageDataGenerator(
#     validation_split=VALIDATION_SPLIT,
#     preprocessing_function=tf.keras.applications.mobilenet_v3.preprocess_input,
# )
#
# train_generator = datagen.flow_from_dataframe(
#     df, x_col='image', y_col='labels',
#     directory=DIRECTORY,
#     target_size=(IMSIZE, IMSIZE),
#     batch_size=BATCH_SIZE, seed=SEED,
#     subset='training',
# )
#
# validation_generator = datagen.flow_from_dataframe(
#     df, x_col='image', y_col='labels',
#     directory=DIRECTORY,
#     target_size=(IMSIZE, IMSIZE),
#     batch_size=BATCH_SIZE, seed=SEED,
#     subset='validation',
# )


# def parse_function(filename, label):
#     image_string = tf.io.read_file(filename)
#     image_decoded = tf.image.decode_jpeg(image_string, channels=3)
#     image_resized = tf.image.resize(image_decoded, [IMSIZE, IMSIZE])
#     image = BACKBONE.preprocess_input(image_resized)
#     return image, label
#
#
# def solve_dataset(df, directory=DIRECTORY,
#                   training=True,
#                   batch_size=BATCH_SIZE, buffer_size=10000):
#     AUTOTUNE = tf.data.experimental.AUTOTUNE
#     filenames = tf.constant([directory + x for x in df['image'].values])
#     labels = tf.constant(df['labels'].values)
#
#     dataset = (
#         tf.data.Dataset
#             .from_tensor_slices((filenames, labels))
#             .map(parse_function, num_parallel_calls=AUTOTUNE)
#             .map(lambda x, y: (RandAugment(x, training=training), y), num_parallel_calls=AUTOTUNE)
#             .shuffle(buffer_size)
#             .repeat()
#             .batch(batch_size)
#             .prefetch(buffer_size=AUTOTUNE)
#     )
#     return dataset


# train_dataset = get_dataset()
# validation_dataset = get_dataset(training=False)


def df_train_test_split(df, test_size=VALIDATION_SPLIT, random_state=SEED):
    train = df.sample(frac=1 - test_size, random_state=random_state)
    test = df.drop(train.index)
    return train, test


def dataset_from_dataframe(df, image_path=IMAGE_PATH,
                           batch_size=BATCH_SIZE,
                           x_col='image', y_col='labels',
                           image_size=IMSIZE, buffer_size=1000,
                           training=True, shuffle=True):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    def parse_function(filename, label):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [image_size, image_size])
        #         image = BACKBONE.preprocess_input(image_resized)[0]
        # one hot encoding the label

        label =
        return image, label

    filenames = [os.path.join(image_path, x) for x in df[x_col].values]
    labels = df[y_col].values

    dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
    dataset = dataset.map(parse_function, num_parallel_calls=AUTOTUNE)

    # dataset = dataset.map(lambda x, y: (BACKBONE.preprocess_input(x), y), num_parallel_calls=AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    if training:
        dataset = dataset.map(lambda x, y: (RandAugment(value_range=(0, 255))(x), y), num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(lambda x, y: (BACKBONE.preprocess_input(x), y), num_parallel_calls=AUTOTUNE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE).repeat()
    return dataset


# df = pd.read_csv(CSV_PATH)
# train_df, test_df = df_train_test_split(df)
#
# train_dataset = dataset_from_dataframe(train_df, shuffle=False)
# validation_dataset = dataset_from_dataframe(test_df, training=False)



def dataset_from_directory(directory, batch_size=BATCH_SIZE,
                           image_size=IMSIZE, buffer_size=10000,
                           training=True, shuffle=True,
                           validation_split=None, seed=SEED, ):
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # def parse_function(image, label):
    #     image = tf.image.resize(image, [image_size, image_size])
    #     return image, label

    if validation_split is None:
        subset = None
    else:
        if training:
            subset = 'training'
        else:
            subset = 'validation'

    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='categorical',
        class_names=None,
        color_mode='rgb',
        batch_size=None,
        image_size=(image_size, image_size),
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset=subset,
        interpolation='bilinear',
        follow_links=False,
    )

    dataset = dataset.map(lambda x, y: (BACKBONE.preprocess_input(x)[0], y), num_parallel_calls=AUTOTUNE)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.repeat().batch(batch_size)
    if training:
        dataset = dataset.map(lambda x, y: (RandAugment(value_range=(0, 255))(x), y), num_parallel_calls=AUTOTUNE)

    dataset = dataset.prefetch(buffer_size=AUTOTUNE)

    return dataset


def solve_dataset():
    if USE_DF:
        df = pd.read_csv(CSV_PATH)
        train_dataset = dataset_from_dataframe(df, training=True)
        validation_dataset = dataset_from_dataframe(df, training=False)
    else:
        train_dataset = dataset_from_directory(TRAIN_IMAGE_PATH, validation_split=0.2)
        validation_dataset = dataset_from_directory(VALIDATION_IMAGE_PATH, training=False)

    return train_dataset, validation_dataset

# train_dataset = dataset_from_directory(IMAGE_PATH + "Train", validation_split=None)
# validation_dataset = dataset_from_directory(IMAGE_PATH + "Test", training=False)
