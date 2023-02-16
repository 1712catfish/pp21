histories = []
scores = []
image_names = np.empty((0,))
predicts = np.empty((0, len(CFG.classes)))

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_f1_score', mode='max',
        patience=CFG.patience[0], restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_f1_score', mode='max',
        patience=CFG.patience[1], min_lr=CFG.min_lr, verbose=2)]

kfold = KFold(n_splits=CFG.folds, shuffle=True, random_state=CFG.seed)
folds = ['fold_0', 'fold_1', 'fold_2', 'fold_3', 'fold_4']

'''
run training loop
'''
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
