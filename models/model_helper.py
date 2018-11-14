import tensorflow as tf
import re
import functools

TOWER_NAME = 'tower'


class ModelBuildTools(object):

    def _activation_summary(self, x):
        """
        概要汇总函数
        :param x: 待保存变量
        :return: None
        """
        # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        # tf.summary.histogram(x.op.name + '/activations', x)
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
            tf.add_to_collection('losses', weight_decay)

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
            tf.summary.scalar(l.op.name + '_raw', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))

        return loss_averages_op

    class _LoggerHook(tf.train.SessionRunHook):
        pass


class ModelTrainTools(object):

    @classmethod
    def classify_loss(cls, logits, labels):
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
    def classify_accuracy(cls, net_out, labels):
        """
        :param net_out:
        :param labels:
        :return: num of predict crrect, num of predict wrong
        """
        # pre = tf.argmax(net_out, axis=1)
        # assert pre.get_shape().as_list() == labels.get_shape().as_list()
        # bool_value = tf.equal(pre, labels)
        # return tf.count_nonzero(bool_value), tf.subtract(labels.get_shape().as_list().count_nonzero(bool_value))
        # return tf.nn.in_top_k(net_out, labels, 1)

        with tf.name_scope('accuracy'): #先使用namescopy看graph 再尝试使用variablescope
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(net_out, 1), labels)
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar('accuracy', accuracy)
            return accuracy


    @classmethod
    def train(cls, total_loss, global_step,  image_nums, FLAGS):
        """
        :param total_loss:
        :param global_step:
        :param image_nums:
        :param FLAGS: with batchsize, epochs, lr, decay_rate, moveing_average_rate, log_histogram
        :return:
        """
        # batch_size, epochs, learning_rate=1e-3,
        #       decay_rate=0.9, log_histograms=True):

        decay_step = int(image_nums // FLAGS.batch_size * FLAGS.num_epochs)
        lr = tf.train.exponential_decay(FLAGS.learning_rate,
                                        global_step,
                                        decay_step,
                                        decay_rate=FLAGS.decay_rate,
                                        staircase=True)
        tf.summary.scalar('learning_rate', lr)
        opt = tf.train.AdamOptimizer(lr)

        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        for l in losses + [total_loss]:
            tf.summary.scalar(l.op.name + '_raw', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))


        # tf.gradients()
        with tf.control_dependencies([loss_averages_op]):
            grads = opt.compute_gradients(total_loss)

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        # if log_histograms:
        #     for var in tf.trainable_variables():
        #         tf.summary.histogram(var.op.name, var)
        variable_averages = tf.train.ExponentialMovingAverage(
            FLAGS.moving_average_decay, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        if FLAGS.log_histograms:
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)

            # Add histograms for gradients.
        if FLAGS.log_histograms:
            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradients', grad)

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')
        return train_op


def use_queue_wrap(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if 'use_queue' not in kwargs:
            raise ValueError('Please use special kwargs not args! use_queue=True NOT True')
        if not kwargs['use_queue']:
            x = func(*args, **kwargs)
        elif kwargs['use_queue']:
            print('Use queue input...')
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            x = func(*args, **kwargs)

            coord.request_stop()
            coord.join(threads)

        print('\nEND')
        return x

    return wrapper


@use_queue_wrap
def train_batch_and_save(FLAGS, train_list=None, feed_dict=None, is_loadmodel=True):
    # train_list = [train_op, total_loss, accuracy]
    # nx_train, ny_train = None, None

    steps = FLAGS.images // FLAGS.batch_size

    sess = tf.Session()
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.output_dir, sess.graph_def)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # split
    checkpoint_dir = tf.train.get_checkpoint_state(FLAGS.output_dir)
    if checkpoint_dir and checkpoint_dir.model_checkpoint_path and is_loadmodel==True:
        saver.restore(sess, checkpoint_dir.model_checkpoint_path)
    else:
        print('Not found checkpoint file or is_loadmodel=False')

    if FLAGS.use_queue == True:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

    # for epoch in range(FLAGS.epochs):
    #     for step in range(steps):
    #
    #         if step and step % FLAGS.log_interval == 0:
    #             解决方法：bx,by, = sess.run([batch_x, batch_y])
    #                         sess.run([], feed_dict={:bx,:by})
    #             _, loss_value, acc_value, summary_str = sess.run(train_list + [summary_op], feed_dict=feed_dict)
    #             logger.info('[{0} / {1}] batch {2}, loss {3}'.format(
    #                 step // log_interval,
    #                 steps // log_interval,
    #                 step, loss_value)
    #             summary_writer.add_summary(summary_str, step)
    #             pass
    #
    # if use_queue == True:
    #     coord.request_stop()
    #     coord.join(threads)

def sess_and_saver_initial(output_dir, is_loadmodel):
    sess = tf.Session()
    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(output_dir, sess.graph)

    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    # split
    checkpoint_dir = tf.train.get_checkpoint_state(output_dir)
    if checkpoint_dir and checkpoint_dir.model_checkpoint_path and is_loadmodel == True:
        saver.restore(sess, checkpoint_dir.model_checkpoint_path)
    else:
        print('Not found checkpoint file or is_loadmodel=False')
    return sess, summary_op, summary_writer, saver

# def train_epochs_and_steps_with_multi_reader()