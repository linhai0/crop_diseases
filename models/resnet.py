import tensorflow as tf
import re
from models.model_helper import ModelBuildTools


class ResNet(object):
    """Resnet model"""

    def __init__(self, config):
        self.mtools = ModelBuildTools()
        self.num_classes = config.num_classes
        # self.batch_size = config.batch_size
        tf.pad()

    tf.nn.conv2d()

    def batch_norm(self, inputs, training=False, data_format=None):
        return tf.layers.batch_normalization(
            inputs=inputs, training=training)

    def conv2d(self, inputs, filters, kernel_size, strides=(1,1), padding='SAME',use_bias=True):
        return tf.layers.conv2d(inputs, filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias)

    def fixed_padding(self, inputs, kernel_size, data_format):
        pad_total = kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg

        if data_format == 'channels_first':
            padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                            [pad_beg, pad_end], [pad_beg, pad_end]])
        else:
            padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                            [pad_beg, pad_end], [0, 0]])
        return padded_inputs

    def conv2d_fixed_padding(self, inputs, filters, kernel_size, strides, data_format, use_bias=False):
        if strides > 1:
            inputs = self.fixed_padding(inputs, kernel_size, data_format)
        return tf.layers.conv2d(inputs, filters=filters, kernel_size=kernel_size,
                                strides=strides, padding=('SAME' if strides == 1 else 'VALID'), use_bias=use_bias,
                                kernel_initializer=tf.variance_scaling_initializer(), data_format=data_format)

    def _building_block_v1(self, inputs, filters, training, projection_shortcut, strides, data_format='channels_last'):
        """
        :param inputs:
        :param filters:
        :param training:
        :param projection_shortcut:projection_shortcut: The function to use for projection shortcuts
           (typically a 1x1 convolution when downsampling the input).
        :param strides:
        :param data_format:
        :return:
        """

        shortcut = inputs
        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs)
            shortcut = self.batch_norm(inputs=shortcut, training=training, data_format=data_format)

        x = self.conv2d_fixed_padding(inputs=inputs, filters=filters, kernel_size=3, strides=strides,
                                           data_format=data_format)
        x = self.batch_norm(x, training)
        x = tf.nn.relu(x)


        x = self.conv2d_fixed_padding(
            inputs=x, filters=filters, kernel_size=3, strides=1,
            data_format=data_format)
        x = self.batch_norm(x, training)


        x += shortcut
        x = tf.nn.relu(x)
        return x

    def _building_block_v2(self, inputs, filters, training, projection_shortcut, strides, data_format):

        shortcut = inputs
        x = self.batch_norm(inputs, training, data_format)
        x = tf.nn.relu(x)

        if projection_shortcut is not None:
            shortcut = projection_shortcut(x)

        x = self.conv2d_fixed_padding(
            inputs=x, filters=filters, kernel_size=3, strides=strides,
            data_format=data_format)

        x = self.batch_norm(x, training, data_format)
        x = tf.nn.relu(x)
        x = self.conv2d_fixed_padding(
            inputs=x, filters=filters, kernel_size=3, strides=1,
            data_format=data_format)
        return x + shortcut

    def _bottleneck_block_v1(self, inputs, filters, training, projection_shortcut,
                             strides, data_format):

        shortcut = inputs

        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs)
            shortcut = self.batch_norm(inputs=shortcut, training=training,
                                  data_format=data_format)

        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=1, strides=1,
            data_format=data_format)
        inputs = self.batch_norm(inputs, training, data_format)
        inputs = tf.nn.relu(inputs)

        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=strides,
            data_format=data_format)
        inputs = self.batch_norm(inputs, training, data_format)
        inputs = tf.nn.relu(inputs)

        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
            data_format=data_format)
        inputs = self.batch_norm(inputs, training, data_format)
        inputs += shortcut
        inputs = tf.nn.relu(inputs)

        return inputs

    def _bottleneck_block_v2(self, inputs, filters, training, projection_shortcut,
                             strides, data_format):

        shortcut = inputs
        inputs = self.batch_norm(inputs, training, data_format)
        inputs = tf.nn.relu(inputs)

        # The projection shortcut should come after the first batch norm and ReLU
        # since it performs a 1x1 convolution.
        if projection_shortcut is not None:
            shortcut = projection_shortcut(inputs)

        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=1, strides=1,
            data_format=data_format)

        inputs = self.batch_norm(inputs, training, data_format)
        inputs = tf.nn.relu(inputs)
        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=filters, kernel_size=3, strides=strides,
            data_format=data_format)

        inputs = self.batch_norm(inputs, training, data_format)
        inputs = tf.nn.relu(inputs)
        inputs = self.conv2d_fixed_padding(
            inputs=inputs, filters=4 * filters, kernel_size=1, strides=1,
            data_format=data_format)

        return inputs + shortcut



