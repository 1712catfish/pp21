try:
    INTERACTIVE
except NameError:
    from NickKuzmenkov.settings import *

NUM_CLASSES = len(Settings.classes)

Settings.model_img_size = 280
Settings.loss = "binary_crossentropy"
Settings.optimizer = "adam"
Settings.metrics = [
    "accuracy",
    tfa.metrics.F1Score(
        num_classes=NUM_CLASSES,
        average='macro'
    )
]


def get_model():
    model = tf.keras.models.Sequential(name='EfficientNetB4')

    model.add(efn.EfficientNetB4(
        include_top=False,
        input_shape=(Settings.model_img_size, Settings.model_img_size, 3),
        weights='noisy-student',
        pooling='avg'))

    model.add(tf.keras.layers.Dense(
        Settings.num_classes,
        kernel_initializer=tf.keras.initializers.RandomUniform(seed=Settings.seed),
        bias_initializer=tf.keras.initializers.Zeros(), name='dense_top')
    )
    model.add(tf.keras.layers.Activation('sigmoid', dtype='float32'))

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer='adam',
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='acc'),
            tfa.metrics.F1Score(
                num_classes=len(Settings.classes),
                average='macro')
        ]
    )

    return model
