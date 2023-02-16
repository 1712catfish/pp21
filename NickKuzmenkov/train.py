try:
    INTERACTIVE
except NameError:
    from NickKuzmenkov.settings import *


def count_data_items(filenames):
    return np.sum([int(x[:-6].split('-')[-1]) for x in filenames])


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.reshape(image, [Settings.tf_record_img_size, Settings.tf_record_img_size, 3])
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
        label = [tf.cast(example[x], tf.float32) for x in Settings.classes]
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
        dataset = dataset.shuffle(2048, seed=Settings.seed)
    if repeated:
        dataset = dataset.repeat()
    dataset = dataset.batch(Settings.batch_size)
    if cached:
        dataset = dataset.cache()
    dataset = dataset.prefetch(auto)
    if distributed:
        dataset = Settings.strategy.experimental_distribute_dataset(dataset)
    return dataset


histories = []
scores = []
image_names = np.empty((0,))
predicts = np.empty((0, len(Settings.classes)))

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_f1_score', mode='max',
        patience=Settings.patience[0], restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_f1_score', mode='max',
        patience=Settings.patience[1], min_lr=Settings.min_lr, verbose=2)]

kfold = KFold(n_splits=Settings.folds, shuffle=True, random_state=Settings.seed)
folds = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']

'''
run training loop
'''
for i, (train_index, val_index) in enumerate(kfold.split(folds)):

    '''
    run only selected folds
    '''
    if i in Settings.used_folds:

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
        with Settings.strategy.scope():
            model = get_model()

        '''
        data setup
        '''
        train_filenames = []
        for j in train_index:
            train_filenames += tf.io.gfile.glob(os.path.join(Settings.gcs_path_aug[0], folds[j], '*.tfrec'))
            train_filenames += tf.io.gfile.glob(os.path.join(Settings.gcs_path_aug[1], folds[j], '*.tfrec'))
            train_filenames += tf.io.gfile.glob(os.path.join(Settings.gcs_path_aug[2], folds[j], '*.tfrec'))
            train_filenames += tf.io.gfile.glob(os.path.join(Settings.gcs_path_aug[3], folds[j], '*.tfrec'))
        np.random.shuffle(train_filenames)

        val_filenames = []
        for j in val_index:
            val_filenames += tf.io.gfile.glob(os.path.join(Settings.gcs_path_raw, folds[j], '*.tfrec'))

        train_dataset = get_dataset(
            train_filenames,
            ordered=False, shuffled=True, repeated=True)

        val_dataset = get_dataset(
            val_filenames,
            cached=True)

        steps_per_epoch = count_data_items(train_filenames) // (20 * Settings.batch_size)
        validation_steps = count_data_items(val_filenames) // Settings.batch_size

        '''
        fit
        '''
        history = model.fit(
            train_dataset,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            callbacks=callbacks,
            epochs=Settings.epochs,
            verbose=Settings.verbose).history

        '''
        write out-of-fold predictions
        '''
        size = count_data_items(val_filenames)
        steps = size // Settings.batch_size + 1

        val_dataset = get_dataset(val_filenames, labeled=False, distributed=False)
        val_predicts = model.predict(
            val_dataset.map(lambda x, y: x),
            steps=steps,
            verbose=Settings.verbose)[:size]
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
