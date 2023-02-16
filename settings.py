import os

from keras_cv_attention_models import *

IMSIZE = 224
SEED = 123
BATCH_SIZE = 32
NUM_CLASSES = 12
VALIDATION_SPLIT = 0.2

# DIRECTORY = "/kaggle/input/plant-pathology-2021-fgvc8/train_images/"
CSV_PATH = "train.csv"
DIRECTORY = "/content/Images"
IMAGE_PATH = "/content/Images/"

BACKBONE = mobilenetv3.MobileNetV3Small(num_classes=0)

USE_DF = True
TRAIN_IMAGE_PATH = None
VALIDATION_IMAGE_PATH = None

STEPS_PER_EPOCH = len(os.listdir(TRAIN_IMAGE_PATH)) // BATCH_SIZE
VALIDATION_STEPS = len(os.listdir(VALIDATION_IMAGE_PATH)) // BATCH_SIZE

