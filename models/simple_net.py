import tensorflow as tf
import re
from models.model_helper import ModelBuildTools


class sim_net(object):

    def __init__(self, config):
        self.mtools = ModelBuildTools()
        self.num_classes = config.num_classes
        # self.batch_size = config.batch_size

    def inference(self, images):
        with tf.variable_scope('conv1') as scope:
            kernel = self.mtools._variable_with_weight_decay('weights', shape=[3, 3, 3, 64],
                                                            stddev=0.01, wd=5e-4)
            bias = self.mtools._variable_on_cpu('bias', [64], tf.constant_initializer(0))
            conv1 = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            x = tf.nn.bias_add(conv1, bias)
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')
            self.mtools._activation_summary(x)

        with tf.variable_scope('conv2') as scope:
            x = tf.contrib.layers.conv2d(x, 128, kernel_size=(3, 3), stride=1, padding='SAME',
                                         scope=scope)
            x = tf.contrib.layers.batch_norm(x, scope=scope.name + '/batch_norm')
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')

        with tf.variable_scope('conv3') as scope:
            x = self.mtools._conv(scope, x, 128, 256, [3, 3])
            x = self.mtools._batchnorm(scope, x)

            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')

        with tf.variable_scope('conv4') as scope:
            x = tf.contrib.layers.conv2d(x, 256, kernel_size=(3, 3), stride=1, padding='SAME',
                                         scope=scope)
            x = tf.contrib.layers.batch_norm(x, scope=scope.name + '/batch_norm')
            x = tf.nn.relu(x)
            x = tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool')

        with tf.variable_scope('dense') as scope:
            shape = x.get_shape().as_list()
            print(shape)
            dim = 1
            for d in shape[1:]:
                dim *= d

            reshape = tf.reshape(x, [-1, dim])
            weights = self.mtools._variable_with_weight_decay('weights', shape=[dim, 1024],
                                                             stddev=0.04, wd=0.004)
            print(weights.get_shape())
            biases = self.mtools._variable_on_cpu('biases', [1024], tf.constant_initializer(0.1))
            x = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
            # x = tf.layers.dense(x,1024, activation='relu')
            self.mtools._activation_summary(x)
            print(x.get_shape())

        with tf.variable_scope('softmax_linear') as scope:
            weights = self.mtools._variable_with_weight_decay('weights', [1024, self.num_classes],
                                                             stddev=1 / 128.0, wd=None)
            biases = self.mtools._variable_on_cpu('biases', [self.num_classes],
                                                 tf.constant_initializer(0.0))
            softmax_linear = tf.add(tf.matmul(x, weights), biases, name=scope.name)
            self.mtools._activation_summary(softmax_linear)

            return softmax_linear

    # def loss(self, net_out, labels):
    #     labels = tf.cast(labels, tf.int64)
    #     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #         labels=labels, logits=net_out, name='cross_entropy')
    #     loss_mean = tf.reduce_mean(cross_entropy)
    #     tf.add_to_collection('losses', loss_mean)
    #     return loss_mean


