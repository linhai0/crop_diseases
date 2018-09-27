from multiprocessing import Process, Queue
import os, time, random
import numpy as np
import tensorflow as tf


# 写数据进程执行的代码:
def write(q,a):
    print('Process to write: %s' % os.getpid())
    for value in list(range(10)):
        x = np.array([value])
        q.put(x)
        print('Put to %s queue...' % value)
        time.sleep(.1)

# 读数据进程执行的代码:
def read(q, sess):
    print('Process to read: %s' % os.getpid())
    for _ in range(10):
        x = q.get(True)
        x_t = sess.run(x_out, feed_dict={x_in:x})
        print('Get from queue.', x_t.shape)
        time.sleep(.1)



x_in = tf.placeholder(dtype=tf.float32, shape=[1])
x_out = tf.reduce_mean(x_in)


def main(q, q1):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())



    for i in range(10):
        x = q.get(True)
        print('Get to', x[0])

        # print(x)
        x_t = sess.run(x_out, feed_dict={x_in: x})
        print('sess run result', x_t)
        time.sleep(.5)
        if i % 2 == 0 and i:
            x1 = q1.get(True)
            print('Get to', x[0])
            x_t = sess.run(x_out, feed_dict={x_in: x1})
            print('sess run result         q1', x_t)

    # 启动子进程pw，写入:
    # 启动子进程pr，读取:
    # pr.start()

    # 等待pw结束:


if __name__=='__main__':
# 父进程创建Queue，并传给各个子进程：

    try :
        q = Queue(2)
        q1 = Queue(2)
        pw = Process(target=write, args=(q,1))
        pw1 = Process(target=write, args=(q1,2))

        # pr = Process(target=read, args=(q,))
        pw.start()
        pw1.start()
        main(q, q1)

        pw.join()
        pw1.terminate()
    except KeyboardInterrupt as e:
        print('KeyboardInter', e)
    finally:
        pw.terminate()
        pw1.terminate()


#     sess = tf.Session()
#     sess.run(tf.global_variables_initializer())
#
#     q = Queue(2)
#     q1 = Queue(2)
#     pw = Process(target=write, args=(q,))
#     pw1 = Process(target=write, args=(q1,))
#
# # pr = Process(target=read, args=(q,))
#     pw.start()
#     pw1.start()
#
#     for i in range(10):
#         x = q.get(True)
#         print('Get to', x[0])
#
#         # print(x)
#         x_t = sess.run(x_out, feed_dict={x_in: x})
#         print('sess run result', x_t)
#         time.sleep(.5)
#         if i % 2 == 0 and i:
#             x1 = q1.get(True)
#             print('Get to', x[0])
#             x_t = sess.run(x_out, feed_dict={x_in: x1})
#             print('sess run result         q1', x_t)
#
# # 启动子进程pw，写入:
#     # 启动子进程pr，读取:
#     # pr.start()
#
#     # 等待pw结束:
#     pw.join()
#     pw1.terminate()

    # pr进程里是死循环，无法等待其结束，只能强行终止:
    # pr.terminate()