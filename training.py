import os
from settings import *


def get_kaggle_gcs():
    from kaggle_datasets import KaggleDatasets
    GCS_DS_PATH = KaggleDatasets().get_gcs_path()
    return GCS_DS_PATH


def train_model(model, train_dataset, validation_dataset, epochs=10, steps_per_epoch=100, validation_steps=100):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    model.fit(
        train_dataset,
        epochs=10,
        steps_per_epoch=,
        validation_data=validation_dataset,
        validation_steps=len(os.listdir(VALIDATION_IMAGE_PATH)) // BATCH_SIZE,
        verbose=1
    )
    return model
