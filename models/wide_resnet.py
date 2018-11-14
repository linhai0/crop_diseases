import tensorflow as tf
import re
from models.model_helper import model_build_tools

import tensorflow as tf
tf.layers.Conv2D
from keras import backend as K
from keras.layers import Dense
from keras.objectives import categorical_crossentropy
from keras.metrics import categorical_accuracy as accuracy
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.layers import Dense, Activation, Flatten, Lambda, Conv2D, AveragePooling2D, \
    BatchNormalization, Dropout, Reshape, Add, Input
from keras.models import Model
from keras.engine.training import collect_trainable_weights
import numpy as np

class sim_net(object):

    def __init__(self, config):
        self.utils = model_build_tools()
        self.num_class = config.num_class
        self.batch_size = config.batch_size



    class CNNEnv:
        def __init__(self):
            # TF fix
            K.set_learning_phase(0)

            # The data, shuffled and split between train and test sets
            (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()

            # Reorder dimensions for tensorflow
            self.x_train = np.transpose(self.x_train.astype('float32'), (0, 1, 2, 3))
            self.mean = np.mean(self.x_train, axis=0, keepdims=True)
            self.std = np.std(self.x_train)
            self.x_train = (self.x_train - self.mean) / self.std
            self.x_test = np.transpose(self.x_test.astype('float32'), (0, 1, 2, 3))
            self.x_test = (self.x_test - self.mean) / self.std

            # self.x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1]*self.x_train.shape[2]*self.x_train.shape[3])
            # self.x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1]*self.x_test.shape[2]*self.x_test.shape[3])
            print('x_train shape:', self.x_train.shape)
            print('x_test shape:', self.x_test.shape)

            # Convert class vectors to binary class matrices
            self.y_train = np_utils.to_categorical(self.y_train)
            self.y_test = np_utils.to_categorical(self.y_test)
            print('y_train shape:', self.y_train.shape)
            print('y_test shape:', self.y_test.shape)

            # For generator
            self.num_examples = self.x_train.shape[0]
            self.index_in_epoch = 0
            self.epochs_completed = 0

            # For wide resnets
            self.blocks_per_group = 4
            self.widening_factor = 4

            # Basic info
            self.batch_num = 64
            self.img_row = 32
            self.img_col = 32
            self.img_channels = 3
            self.nb_classes = 10

        def next_batch(self, batch_size):
            """Return the next `batch_size` examples from this data set."""
            self.batch_size = batch_size

            start = self.index_in_epoch
            self.index_in_epoch += self.batch_size

            if self.index_in_epoch > self.num_examples:
                # Finished epoch
                self.epochs_completed += 1
                # Shuffle the data
                perm = np.arange(self.num_examples)
                np.random.shuffle(perm)
                self.x_train = self.x_train[perm]
                self.y_train = self.y_train[perm]

                # Start next epoch
                start = 0
                self.index_in_epoch = self.batch_size
                assert self.batch_size <= self.num_examples
            end = self.index_in_epoch
            return self.x_train[start:end], self.y_train[start:end]

        def step(self):

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            K.set_session(sess)

            def zero_pad_channels(x, pad=0):
                """
                Function for Lambda layer
                """
                pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
                return tf.pad(x, pattern)

            def residual_block(x, nb_filters=16, subsample_factor=1):
                prev_nb_channels = K.int_shape(x)[3]

                if subsample_factor > 1:
                    subsample = (subsample_factor, subsample_factor)
                    # shortcut: subsample + zero-pad channel dim
                    shortcut = AveragePooling2D(pool_size=subsample)(x)
                else:
                    subsample = (1, 1)
                    # shortcut: identity
                    shortcut = x

                if nb_filters > prev_nb_channels:
                    shortcut = Lambda(zero_pad_channels,
                                      arguments={
                                          'pad': nb_filters - prev_nb_channels})(shortcut)

                y = BatchNormalization(axis=3)(x)
                y = Activation('relu')(y)
                y = Conv2D(nb_filters, 3, 3, subsample=subsample,
                                  init='he_normal', border_mode='same')(y)
                y = BatchNormalization(axis=3)(y)
                y = Activation('relu')(y)
                y = Conv2D(nb_filters, 3, 3, subsample=(1, 1),
                                  init='he_normal', border_mode='same')(y)

                out = Add()([y, shortcut])

                return out

            # this placeholder will contain our input digits
            img = tf.placeholder(tf.float32, shape=(None, self.img_col, self.img_row, self.img_channels))
            labels = tf.placeholder(tf.float32, shape=(None, self.nb_classes))
            # img = K.placeholder(ndim=4)
            # labels = K.placeholder(ndim=1)

            # Keras layers can be called on TensorFlow tensors:
            x = Conv2D(16, 3, 3, init='he_normal', border_mode='same')(img)

            for i in range(0, self.blocks_per_group):
                nb_filters = 16 * self.widening_factor
                x = residual_block(x, nb_filters=nb_filters, subsample_factor=1)

            for i in range(0, self.blocks_per_group):
                nb_filters = 32 * self.widening_factor
                if i == 0:
                    subsample_factor = 2
                else:
                    subsample_factor = 1
                x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)

            for i in range(0, self.blocks_per_group):
                nb_filters = 64 * self.widening_factor
                if i == 0:
                    subsample_factor = 2
                else:
                    subsample_factor = 1
                x = residual_block(x, nb_filters=nb_filters, subsample_factor=subsample_factor)

            x = BatchNormalization(axis=3)(x)
            x = Activation('relu')(x)
            x = AveragePooling2D(pool_size=(8, 8), strides=None, border_mode='valid')(x)
            x = tf.reshape(x, [-1, np.prod(x.get_shape()[1:].as_list())])

            # Readout layer
            preds = Dense(self.nb_classes, activation='softmax')(x)

            loss = tf.reduce_mean(categorical_crossentropy(labels, preds))

            optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

            '''
            with sess.as_default():

                for i in range(10):

                    batch = self.next_batch(self.batch_num)
                    _, l = sess.run([optimizer, loss],
                                    feed_dict={img: batch[0], labels: batch[1]})
                    print(l)
            '''

            with sess.as_default():
                batch = self.next_batch(self.batch_num)
                _, l = sess.run([optimizer, loss],
                                feed_dict={img: batch[0], labels: batch[1]})
                print('Loss', l)

            acc_value = accuracy(labels, preds)

            '''
            with sess.as_default():
                acc = acc_value.eval(feed_dict={img: self.x_test, labels: self.y_test})
                print(acc)
            '''

    a = CNNEnv()
    a.step()


import keras
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, Flatten, AveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.regularizers import l2
from keras import optimizers
from keras.models import Model

DEPTH = 28
WIDE = 10
IN_FILTERS = 16

CLASS_NUM = 10
IMG_ROWS, IMG_COLS = 32, 32
IMG_CHANNELS = 3

BATCH_SIZE = 128
EPOCHS = 200
ITERATIONS = 50000 // BATCH_SIZE + 1
WEIGHT_DECAY = 0.0005
LOG_FILE_PATH = './w_resnet/'

from keras import backend as K

# set GPU memory
if ('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


def scheduler(epoch):
    if epoch < 60:
        return 0.1
    if epoch < 120:
        return 0.02
    if epoch < 160:
        return 0.004
    return 0.0008


def color_preprocessing(x_train, x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    mean = [125.3, 123.0, 113.9]
    std = [63.0, 62.1, 66.7]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]

    return x_train, x_test


def wide_residual_network(img_input, classes_num, depth, k):
    print('Wide-Resnet %dx%d' % (depth, k))
    n_filters = [16, 16 * k, 32 * k, 64 * k]
    n_stack = (depth - 4) // 6

    def conv3x3(x, filters):
        return Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=l2(WEIGHT_DECAY),
                      use_bias=False)(x)

    def bn_relu(x):
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        return x

    def residual_block(x, out_filters, increase=False):
        global IN_FILTERS
        stride = (1, 1)
        if increase:
            stride = (2, 2)

        o1 = bn_relu(x)

        conv_1 = Conv2D(out_filters,
                        kernel_size=(3, 3), strides=stride, padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(WEIGHT_DECAY),
                        use_bias=False)(o1)

        o2 = bn_relu(conv_1)

        conv_2 = Conv2D(out_filters,
                        kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_initializer='he_normal',
                        kernel_regularizer=l2(WEIGHT_DECAY),
                        use_bias=False)(o2)
        if increase or IN_FILTERS != out_filters:
            proj = Conv2D(out_filters,
                          kernel_size=(1, 1), strides=stride, padding='same',
                          kernel_initializer='he_normal',
                          kernel_regularizer=l2(WEIGHT_DECAY),
                          use_bias=False)(o1)
            block = add([conv_2, proj])
        else:
            block = add([conv_2, x])
        return block

    def wide_residual_layer(x, out_filters, increase=False):
        global IN_FILTERS
        x = residual_block(x, out_filters, increase)
        IN_FILTERS = out_filters
        for _ in range(1, int(n_stack)):
            x = residual_block(x, out_filters)
        return x

    x = conv3x3(img_input, n_filters[0])
    x = wide_residual_layer(x, n_filters[1])
    x = wide_residual_layer(x, n_filters[2], increase=True)
    x = wide_residual_layer(x, n_filters[3], increase=True)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((8, 8))(x)
    x = Flatten()(x)
    x = Dense(classes_num,
              activation='softmax',
              kernel_initializer='he_normal',
              kernel_regularizer=l2(WEIGHT_DECAY),
              use_bias=False)(x)
    return x


if __name__ == '__main__':
    # load data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    y_train = keras.utils.to_categorical(y_train, CLASS_NUM)
    y_test = keras.utils.to_categorical(y_test, CLASS_NUM)

    # color preprocessing
    x_train, x_test = color_preprocessing(x_train, x_test)

    # build network
    img_input = Input(shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS))
    output = wide_residual_network(img_input, CLASS_NUM, DEPTH, WIDE)
    resnet = Model(img_input, output)
    print(resnet.summary())
    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    resnet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # set callback
    tb_cb = TensorBoard(log_dir=LOG_FILE_PATH, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    cbks = [change_lr, tb_cb]

    # set data augmentation
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 width_shift_range=0.125,
                                 height_shift_range=0.125,
                                 rotation_range=5,
                                 vertical_flip=True, fill_mode='reflect')

    # datagen.fit(x_train)

    # start training
    resnet.fit_generator(datagen.flow(x_train, y_train,
                                      batch_size=BATCH_SIZE),
                         steps_per_epoch=ITERATIONS,
                         epochs=EPOCHS,
                         callbacks=cbks,
                         validation_data=(x_test, y_test))
    # resnet.save('wresnet_re_{:d}x{:d}.h5'.format(DEPTH,WIDE))
    resnet.save_weights('wresnet_{:d}x{:d}_w4_augment.h5'.format(DEPTH, WIDE))


