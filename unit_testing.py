import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import glob

data_dir = '/home/linhai/Pictures'

image_list = glob.glob(data_dir + '/*')
label_list = np.arange(len(image_list))

def abc():
    xx = 1
    def a():
        print('a',xx)

    def b(x=xx):
        print('b',x)
    return a