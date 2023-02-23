import tensorflow as tf


def solve_hardware():
    tf.random.set_seed(42)
    print('Using tensorflow %s' % tf.__version__)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')

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

    return strategy
