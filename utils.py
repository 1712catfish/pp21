import tensorflow as tf


def solve_hardware():
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("Device:", tpu.master())
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    except:
        strategy = tf.distribute.get_strategy()
    print('#Replicas: ', strategy.num_replicas_in_sync)

    return strategy


