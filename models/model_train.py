import tensorflow as tf
from multiprocessing import Queue, Process
# from config import FLAGS
from config import random_transform_args
from data.get_train_data import gen_train_data, multi_process_data_read
# from models.simple_net import sim_net
from models.model_helper import ModelTrainTools as train_tools
import models.model_helper as MH
import numpy as np
import os, sys, random
from tools.utils import get_logger


# import data.get_train_data as get


# g1 = gen_train()
# g2 = val_train()


def multi_train(Net, FLAGS):
    with tf.Graph().as_default():

        global_step = tf.Variable(0, trainable=False)

        train_func, valid_func, [train_steps, valid_steps] = multi_process_data_read(FLAGS)

        images = tf.placeholder(dtype=tf.float32, shape=[None] + FLAGS.image_size + [3], name='input')
        labels = tf.placeholder(dtype=tf.int64, shape=[None], name='labels')

        logits = Net.inference(images)
        total_loss = train_tools.classify_loss(logits, labels)
        train_op = train_tools.train(total_loss, global_step, 0.9,
                                     FLAGS.image_nums,
                                     FLAGS.batch_size,
                                     FLAGS.epochs)
        accuracy_op = train_tools.classify_accuracy(logits, labels)
        best_model0 = 0

        # sess define, load model, model initial
        sess, summary_op, summary_writer, saver = MH.sess_and_saver_initial(FLAGS.output_dir, FLAGS.is_loadmodel)
        logger = get_logger(FLAGS.model_name, FLAGS.output_dir)

        train_queue = Queue(2)
        valid_queue = Queue(2)
        train_process = Process(target=train_func, args=(train_queue,))
        valid_process = Process(target=valid_func, args=(valid_queue,))

        try:
            train_process.start()
            valid_process.start()
            main(train_queue, valid_queue)
            train_process.join()
            valid_process.terminate()
        except KeyboardInterrupt as e:
            print('KeyboardInter', e)
        finally:
            train_process.terminate()
            valid_process.terminate()

        for epoch in range(FLAGS.epochs):
            for step in range(train_steps):
                batch_x, batch_y = train_queue.get(True)
                if step and step % FLAGS.log_interval == 0:
                    _, loss_value, acc_value, summary_str = sess.run([train_op, total_loss, accuracy_op, summary_op],
                                                                     feed_dict={images: batch_x, labels: batch_y})

                    logger.info('[{0} / {1}] batch {2}, loss {3}'.format(
                        step // FLAGS.log_interval,
                        valid_steps // FLAGS.log_interval,
                        step, loss_value))
                    summary_writer.add_summary(summary_str, global_step=global_step)
                else:
                    _ = sess.run([train_op], feed_dict={images: batch_x, labels: batch_y})

            best_model0, best_model1 = 0,1
            accuracy_sum = []
            for step in range(valid_steps):
                batch_x, batch_y = valid_queue.get(True)
                if step and step % FLAGS.log_interval == 0:
                    _, loss_value, acc_value = sess.run([total_loss, accuracy_op],
                                                                     feed_dict={images: batch_x, labels: batch_y})

                    logger.info('[{0} / {1}] batch {2}, loss {3}'.format(
                        step // FLAGS.log_interval,
                        valid_steps // FLAGS.log_interval,
                        step, loss_value))
                    accuracy_sum.append(acc_value)
                    # summary_writer.add_summary(summary_str, step)
                else:
                    acc_value = sess.run([accuracy_op], feed_dict={images: batch_x, labels: batch_y})
                    accuracy_sum.append(acc_value)

            predicted_acc = np.mean(accuracy_sum)
            logger.info('#'*40+'\n'+'#'*40+'\t'+'ACC: ',predicted_acc,'#'*40+'\n'+'#'*40)
            summary = tf.Summary()
            summary.ParseFromString(summary_str)
            summary.value.add(tag='val_acc', simple_value=predicted_acc)
            summary_writer.add_summary(summary, epoch)
            if predicted_acc > best_model0:
                best_model0 = predicted_acc
            checkpoint_path = os.path.join(FLAGS.output_dir, FLAGS.model_name + '.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)




def train(Net, FLAGS):
    with tf.Graph().as_default():

        global_step = tf.Variable(0, trainable=False)

        gen_train, val_train = gen_train_data(
            FLAGS)

        images = tf.placeholder(dtype=tf.float32, shape=[None] + FLAGS.image_size + [3], name='input')
        labels = tf.placeholder(dtype=tf.int64, shape=[None], name='labels')

        logits = Net.inference(images)
        total_loss = train_tools.classify_loss(logits, labels)
        train_op = train_tools.train(total_loss, global_step, 0.9,
                                     FLAGS.image_nums,
                                     FLAGS.batch_size,
                                     FLAGS.epochs)
        acc_op = train_tools.classify_accuracy(logits, labels)
        best_model0 = 0

        # split

        sess = tf.Session()
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.output_dir, sess.graph_def)

        initial = tf.global_variables_initializer()
        sess.run(initial)
        # split
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
                acc_out = sess.run(acc_op, feed_dict={images: batch_images, labels: batch_labels})
                sum_acc.append(acc_out)
            total_acc = np.array(sum_acc).mean()
            print('acc:', total_acc)
            print('\n\n\n\n')
            if total_acc > best_model0:
                best_model0 = total_acc
                checkpoint_path = os.path.join(FLAGS.output_dir, FLAGS.model_name + '.ckpt')
                saver.save(sess, checkpoint_path, global_step=global_step)


