import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# tf.nn.softmax_cross_entropy_with_logits
import glob

data_dir = '/home/linhai/Pictures'

image_list = glob.glob(data_dir + '/*')
label_list = np.arange(len(image_list))


def abc():
    xx = 1

    def a():
        print('a', xx)

    def b(x=xx):
        print('b', x)

    return a


np.random.seed(42)

from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
seed = 42

x_train = (x_train - x_train.mean()) / x_train.std()

x_test = (x_test - x_test.mean()) / x_test.std()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


def mnist_net(x):
    x = tf.keras.layers.Dense(512, activation='relu',
                              # kernel_initializer=tf.keras.initializers.random_normal(stddev=.01, seed=seed),
                              use_bias=True)(x)
    x = tf.keras.layers.Dense(512, activation='relu',
                              # kernel_initializer=tf.keras.initializers.random_normal(stddev=.01, seed=seed),
                              use_bias=True)(x)
    x = tf.keras.layers.Dense(512, activation='relu',
                              # kernel_initializer=tf.keras.initializers.random_normal(stddev=.01, seed=seed),
                              use_bias=True)(x)
    x = tf.keras.layers.Dense(10, activation='softmax')(x)
    return x


in_1 = tf.keras.layers.Input(shape=[28 * 28])
in_2 = tf.keras.layers.Input(shape=[28 * 28])
model_1 = tf.keras.Model(in_1, mnist_net(in_1))
model_2 = tf.keras.Model(in_2, mnist_net(in_2))
model_3 = tf.keras.Model(in_2, mnist_net(in_2))

adm = tf.keras.optimizers.Adam()

model_1.compile(optimizer='SGD',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model_2.compile(optimizer='SGD',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model_3.compile(optimizer='SGD',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model_1.fit(x_train, y_train, batch_size=128, epochs=200, validation_data=(x_test, y_test), verbose=2)
model_2.fit(-x_train, y_train, batch_size=128, epochs=200, validation_data=(-x_test, y_test), verbose=2)
model_3.fit(x_train, y_train, batch_size=128, epochs=200, validation_data=(x_test, y_test), verbose=2)

model_1.save('output/model_1.h5')
model_2.save('output/model_2.h5')
model_3.save('output/model_3.h5')


def dis_layers(model1, model2, model3):
    L = []
    for i in range(1, 5):
        w1, w2, w3 = model1.layers[i].get_weights()[0], model2.layers[i].get_weights()[0], \
                     model3.layers[i].get_weights()[0]
        L.append([np.mean(np.abs(w1 - w2)), np.mean(np.abs(w1 - w3)), np.mean(np.abs(w2 - w3)),
                  np.mean(np.abs(w1 + w2)), np.mean(np.abs(w3 + w1)), np.mean(np.abs(w2 + w3))])
    return L


dis = dis_layers(model_1, model_2, model_3)

'''mnist200e[[0.19473438, 0.16636175, 0.15127374, 0.16982384, 0.1294567, 0.188426],
 [0.17610644, 0.17250572, 0.1302776, 0.18197505, 0.13532744, 0.1782773],
 [0.17261584, 0.17264003, 0.16818987, 0.17754135, 0.17219822, 0.17858993],
 [0.18092509, 0.18342316, 0.19332393, 0.2533417, 0.27755773, 0.25640765]]'''
'''fationmnist50e[[0.25215197, 0.2155574, 0.23031361, 0.22229494, 0.2470201, 0.19213074],
 [0.1953832, 0.19216855, 0.15630218, 0.19890083, 0.1958204, 0.15977071],
 [0.24705313, 0.2542334, 0.24667928, 0.25054058, 0.25783548, 0.24817243],
 [0.22593585, 0.22631426, 0.25039095, 0.3053473, 0.30698106, 0.32472268]]'''
'''200e[[0.04801664, 0.04801874, 0.048075765, 0.04804845, 0.048126206, 0.048080154],
 [0.054203812, 0.05412479, 0.054158833, 0.054118585, 0.054144032, 0.0542343],
 [0.053732105, 0.05391748, 0.053970437, 0.05382757, 0.053822808, 0.05383958],
 [0.13715461, 0.138257, 0.13851266, 0.13742629, 0.13608842, 0.13558957]]
'''
p1 = model_1.predict(x_test)
p2 = model_2.predict(-x_test)
p3 = model_3.predict(x_test)


def concat_acc(p1, p2):
    return np.sum(p1.argmax(axis=1) == p2.argmax(axis=1))


acc = concat_acc(p1, y_test), concat_acc(p2, y_test), concat_acc(p3, y_test), \
      concat_acc(p2 + p1, y_test), concat_acc(p1 + p3, y_test), concat_acc(p2 + p3, y_test)
''' (9821, 9808, 9835, 9820, 9828, 9841) (8947, 8905, 8923, 9048, 9048, 9039, 9049)'''
