import tensorflow as tf
import re
import keras
from keras.layers import BatchNormalization

TOWER_NAME = 'tower'


class model_build_tools(object):

    def _activation_summary(self, x):
        """
        概要汇总函数
        :param x: 待保存变量
        :return: None
        """
        # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tf.summary.histogram(x.op.name + '/activations', x)
        tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x))

    def _variable_on_cpu(self, name, shape, initializer):
        """
        创建变量
        :param name: name_scope
        :param shape: tensor维度
        :param initializer: 初始化值
        :return: tensor变量
        """
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer)

        return var

    def _variable_with_weight_decay(self, name, shape, stddev, wd=None):
        """
        创建有权重衰减项的变量
        :param name: name_scope
        :param shape: tensor维度
        :param stddev: 用于初始化的标准差
        :param wd: 权重
        :return: tensor变量
        """
        # wd 为衰减因子,若为None则无衰减项
        var = self._variable_on_cpu(name, shape, tf.random_normal_initializer(stddev=stddev))
        if wd:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('regu_losses', weight_decay)

        return var

    def _conv(self, scope, x, in_dim, out_dim, kernel_size, stddev=0.01, wd=5e-4, use_bias=True, padding='SAME'):
        kernel = self._variable_with_weight_decay('weights', shape=kernel_size + [in_dim, out_dim],
                                                  stddev=stddev, wd=wd)
        x = tf.nn.conv2d(x, kernel, [1, 1, 1, 1], padding=padding, name=scope.name)
        if use_bias:
            bias = self._variable_on_cpu('biases', [out_dim], tf.constant_initializer(0))
            x = tf.nn.bias_add(x, bias)
        return x

    def _batchnorm(self, scope, x, epsilon=1e-5, momentum=0.9):

        return tf.contrib.layers.batch_norm(x, decay=momentum, updates_collections=None,
                                            epsilon=epsilon, scale=True, scope=scope.name + 'batch_norm')

    def _add_loss_summaries(self, total_loss):
        """
        增加损失概要信息
        :param total_loss:
        :return:
        """
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        for l in losses + [total_loss]:
            tf.summary.scalar(l.op.name + ' (raw)', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))

        return loss_averages_op

    class _LoggerHook(tf.train.SessionRunHook):
        pass


class ModelTrainTools(object):

    @classmethod
    def loss(cls, logits, labels):
        """Add L2Loss to all the trainable variables.
        Add summary for "Loss" and "Loss/avg".
        Args:
          logits: Logits from inference().
          labels: Labels from distorted_inputs or inputs(). 1-D tensor
                  of shape [batch_size]
        Returns:
          Loss tensor of type float.
        """
        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    @classmethod
    def acc(cls, net_out, labels):
        """
        :param net_out:
        :param labels:
        :return: num of predict crrect, num of predict wrong
        """
        # pre = tf.argmax(net_out, axis=0)
        # assert pre.get_shape().as_list() == labels.get_shape().as_list()
        # bool_value = tf.equal(pre, labels)
        # return tf.count_nonzero(bool_value), tf.subtract(labels.get_shape().as_list().count_nonzero(bool_value))
        return tf.nn.in_top_k(net_out, labels, 1)

    @classmethod
    def train(cls, total_loss, global_step, moving_average_decay, image_nums, batch_size, epochs, learning_rate=1e-3,
              decay_rate=0.9, log_histograms=True):
        ###改成传入更少的参数
        """
        :param total_loss:
        :param gloabl_step:
        :param image_nums:
        :param batch_size:
        :param epochs:
        :param learning_rate:
        :param decay_rate:
        :return:
        """
        decay_step = int(image_nums // batch_size * epochs)
        lr = tf.train.exponential_decay(learning_rate,
                                        global_step,
                                        decay_step,
                                        decay_rate=decay_rate,
                                        staircase=True)
        tf.summary.scalar('learning_rate', lr)
        opt = tf.train.AdamOptimizer(lr)

        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        for l in losses + [total_loss]:
            tf.summary.scalar(l.op.name + ' (raw)', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))

        # loss_averages_op = self.utils._add_loss_summaries(total_loss)

        with tf.control_dependencies([loss_averages_op]):
            grads = opt.compute_gradients(total_loss)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        # if log_histograms:
        #     for var in tf.trainable_variables():
        #         tf.summary.histogram(var.op.name, var)
        variable_averages = tf.train.ExponentialMovingAverage(
            moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        if log_histograms:
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

            # Add histograms for gradients.
        if log_histograms:
            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradients', grad)

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')
        return train_op
