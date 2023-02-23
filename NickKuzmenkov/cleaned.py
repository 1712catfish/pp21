# def filter_valid_args(func, args_dict):
#     """Return dictionary without invalid function arguments."""
#     validArgs = func.func_code.co_varnames[:func.func_code.co_argcount]
#     return dict((key, value) for key, value in args_dict.iteritems()
#                 if key in validArgs)




def list_variables(condition=None, filter_underscore=True):
    def cond(k):
        if filter_underscore and k.startswith('__'):
            return False
        if condition is not None:
            return condition(k)

    return {k: eval(k) for k in dir() if cond(k)}




def kfold_index_generator():
    kfolds = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED))
    for index, (train_ids, val_ids) in enumerate(kfolds.split(list(range(FOLDS))):
        (train_dataset, train_size), (val_dataset, val_size) = solve_dataset(train_ids, val_ids)

        yield dict(
            index=index,
            train_ids=train_ids,
            val_ids=val_ids,
            train_data=train_dataset,
            train_size=train_size,
            validation_data=val_dataset,
            val_size=val_size,
            steps_per_epoch=train_size // BATCH_SIZE // 20,
            validation_steps=val_size // BATCH_SIZE,
        )


def train(model, dict_data_generator, **kwargs):
    history = []

    for d in dict_data_generator():
        print(f"========== Fold {d['index']} ==========")

        kwargs = list_constants()


        hist = model.fit(
            d["train_dataset"],
            callbacks=CALLBACKS,
            epochs=EPOCHS,
            verbose=VERBOSE,
            **filter_valid_args(MODEL.fit, d)
        )

        model.save_weights(f'model_{i}.h5')
        history.append(hist.history)
