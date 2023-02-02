from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

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
