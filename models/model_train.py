import tensorflow as tf

# from config import FLAGS
from config import random_transform_args
from data.get_train_data import gen_train_data
# from models.simple_net import sim_net
from models.model_helper import ModelTrainTools as train_tools
import numpy as np
import os, sys, random


# import data.get_train_data as get


# g1 = gen_train()
# g2 = val_train()


def train(Net, FLAGS):
    with tf.Graph().as_default():

        global_step = tf.Variable(0, trainable=False)
        # def optimzer(self, lr=1e-3):
        # opt = tf.train.AdamOptimizer(FLAGS.lr)

        gen_train, val_train = gen_train_data(
            FLAGS)

        images = tf.placeholder(dtype=tf.float32, shape=[None] + FLAGS.image_size + [3], name='input')
        labels = tf.placeholder(dtype=tf.int64, shape=[None], name='labels')

        logits = Net.inference(images)
        total_loss = train_tools.loss(logits, labels)
        train_op = train_tools.train(total_loss, global_step, 0.9,
                             FLAGS.image_nums,
                             FLAGS.batch_size,
                             FLAGS.epochs)
        acc = train_tools.acc(logits, labels)
        best_model0 = 0

        sess = tf.Session()
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.output_dir, sess.graph_def)

        initial = tf.global_variables_initializer()
        sess.run(initial)

        checkpoint_dir = tf.train.get_checkpoint_state(FLAGS.output_dir)
        if checkpoint_dir and checkpoint_dir.model_checkpoint_path:
            saver.restore(sess, checkpoint_dir.model_checkpoint_path)
        else:
            print('Not found checkpoint file')

        for epoch in range(FLAGS.epochs):
            step = 0
            for batch_data in gen_train():
                step += 1
                batch_images, batch_labels = batch_data
                if step % 50 == 0:
                    _, loss_value, summary_str = sess.run([train_op, total_loss, summary_op],
                                                          feed_dict={images: batch_images, labels: batch_labels})

                    print('[epoch:{0} step:{1}] loss:{2}'.format(epoch, step, loss_value))  # 随后更改log版本

                    # summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, step)
                else:
                    _, loss_value = sess.run([train_op, total_loss],
                                             feed_dict={images: batch_images, labels: batch_labels})
                    print('[epoch:{0} step:{1}] loss:{2}'.format(epoch, step, loss_value))  # 随后更改log版本

            print('vildation:')
            sum_acc = []
            for batch_data in val_train():
                batch_images, batch_labels = batch_data
                acc_out = sess.run(acc, feed_dict={images: batch_images, labels: batch_labels})
                sum_acc.append(acc_out)
            total_acc = np.array(sum_acc).mean()
            print('acc:', total_acc)
            print('\n\n\n\n')
            if total_acc > best_model0:
                best_model0 = total_acc
                checkpoint_path = os.path.join(FLAGS.output_dir, FLAGS.model_name + '.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step)
