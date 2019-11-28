import tensorflow as tf


class DQN:
    """
    Create network architecture based on the number of possible actions.
    """
    def __init__(self, n_actions):
        self.n_actions = n_actions

    def create_model(self, state, name):
        with tf.variable_scope(name) as scope:
            normal = tf.divide(
                x=state,
                y=256.0)

            conv_1 = tf.layers.conv2d(
                inputs=normal,
                filters=32,
                kernel_size=8,
                strides=4,
                data_format='channels_first',
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer())

            conv_2 = tf.layers.conv2d(
                inputs=conv_1,
                filters=64,
                kernel_size=4,
                strides=2,
                data_format='channels_first',
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer())

            conv_3 = tf.layers.conv2d(
                inputs=conv_2,
                filters=64,
                kernel_size=3,
                strides=1,
                data_format='channels_first',
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer())

            conv_F = tf.reshape(
                tensor=conv_3,
                shape=[-1, 7*7*64])

            hidden = tf.layers.dense(
                inputs=conv_F,
                units=512,
                activation=tf.nn.relu,
                kernel_initializer=tf.contrib.layers.xavier_initializer())

            qValue = tf.layers.dense(
                inputs=hidden,
                units=self.n_actions,
                use_bias = False,
                kernel_initializer=tf.contrib.layers.xavier_initializer())

        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name)
        return qValue, trainable_vars
