def create_tf_dataset(df, root):
    df['image_path'] = df['image_id'].apply(lambda x: os.path.join(root, x))
    df['label'] = df['label'].astype(str)
    return tf.data.Dataset.from_tensor_slices(dict(df))
