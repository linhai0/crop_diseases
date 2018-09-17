import tensorflow as tf
import numpy as np



import glob
data_dir = '/home/linhai/Pictures'

image_list = glob.glob(data_dir+'/*')
label_list = np.arange(len(image_list))

print(len(image_list))

image_list_tensor = tf.convert_to_tensor(image_list)
label_list_tensor = tf.convert_to_tensor(label_list)

input_queue = tf.train.slice_input_producer([image_list_tensor,label_list_tensor], num_epochs=1, shuffle=False)  #这里有个num_epochs=1的难题需要测试
reader = tf.read_file(input_queue[0])
image = tf.image.decode_jpeg(reader)
print(image.shape)
image = tf.image.resize_images(image, (256,256))
print(image.shape)

with tf.Session() as sess:
    coord = tf.train.Coordinator()

    sess.run([image, input_queue[1]])
    sess.run([image, input_queue[1]])

    sess.run([image, input_queue[1]])
    sess.run([image, input_queue[1]])


# for d in train_list:
#     nx_train.append(d['image'])
#     ny_train.append(d['label'])




# x = np.random.normal(0, 3, size=(1000, 10)).astype(np.float32)
# with tf.Graph().as_default():
#     input_x = tf.placeholder(tf.float32, shape=[None, 10])
#     with tf.name_scope('mutilply') as scope:
#         w1 = tf.get_variable('w1', shape=[5, 10])
#         b1 = tf.get_variable('b1', shape=[10])
#         out1 = tf.multiply(input_x, w1)
#         tf.summary.scalar('out1', tf.reduce_mean(out1))
#         tf.summary.histogram('out1/his', out1)
#         out2 = tf.nn.bias_add(out1, b1)
#     # tf.summary.scalar('out2', tf.reduce_mean(out2))
#     # tf.summary.histogram('out2', out2)
#     sess = tf.Session()
#     merged = tf.summary.merge_all()
#
#     sess.run(tf.global_variables_initializer())
#     summary_writer = tf.summary.FileWriter('output', graph=sess.graph)
#     for i in range(1000 // 5):
#         data = x[i * 5:(i + 1) * 5]
#         print(data.shape)
#         _ = sess.run([out2], feed_dict={input_x: data})
#         mer = sess.run(merged)
#         summary_writer.add_summary(mer, global_step=i)
from keras.preprocessing.image import ImageDataGenerator
ImageDataGenerator()