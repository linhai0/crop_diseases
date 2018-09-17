import os, json, time, random, sys
import tensorflow as tf

# I personally always like to make my paths absolute
# to be independent from where the python binary is called
dir = os.path.dirname(os.path.realpath(__file__))
dataset_dir = '/home/linhai/PycharmProjects/crop_diseases_recognition_dataset'
random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.4,
}

# I won't dig into TF interaction with the shell, feel free to explore the documentation

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


# def get_flags():
del_all_flags(tf.flags.FLAGS)
FLAGS = tf.app.flags.FLAGS
# tf.summary.histogram
# tf.flags
# Hyper-parameters search configuration
tf.flags.DEFINE_boolean('fullsearch', False,
                     'Perform a full search of hyperparameter space ex:(hyperband -> lr search -> hyperband with best lr)')

# Agent configuration
tf.flags.DEFINE_string('model_name', 'sim_net', 'Unique name of the model')
#暂时用不到
# tf.flags.DEFINE_boolean('best', False, 'Force to use the best known configuration')
# tf.flags.DEFINE_float('initial_mean', 0., 'Initial mean for NN')
# tf.flags.DEFINE_float('initial_stddev', 1e-3, 'Initial standard deviation for NN')
# tf.flags.DEFINE_float('lr', 1e-3, 'The learning rate of SGD/ADM')
# tf.flags.DEFINE_float('nb_units', 20, 'Number of hidden units in Deep learning agents')

# # Environment configuration
# tf.flags.DEFINE_boolean('debug', False, 'Debug mode')
tf.flags.DEFINE_integer('max_iter', 2000, 'Number of training step')
tf.flags.DEFINE_boolean('infer', False, 'Load an agent for playing')

tf.flags.DEFINE_boolean('use_imbalance', False, 'whether use imbalance to split dataset include train and validate')

# This is very important for TensorBoard
# each model will end up in its own unique folder using time module
# Obviously one can also choose to name the output folder
tf.flags.DEFINE_string('output_dir', dir + '/output/' + FLAGS.model_name + '/',  # + str(int(time.time())),
                    'Name of the directory to store/log the model (if it exists, the model will be loaded from it)')
tf.flags.DEFINE_string('log_dir', dir+'/output/log_dir/', 'log dir')

# Another important point, you must provide an access to the random seed
# to be able to fully reproduce an experiment
tf.flags.DEFINE_integer('random_seed', random.randint(0, sys.maxsize), 'Value of random seed')

tf.flags.DEFINE_string('train_json_path',
                    dataset_dir + '/AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json',
                    'train dataset json path')
tf.flags.DEFINE_string('validate_json_path',
                    dataset_dir + '/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json',
                    'validate dataset json path')
tf.flags.DEFINE_string('train_data_dir',
                    dataset_dir + '/AgriculturalDisease_trainingset/images',
                    'train data dir')
tf.flags.DEFINE_string('validate_data_dir',
                    dataset_dir + '/AgriculturalDisease_validationset/images',
                    'validate data dir')
tf.flags.DEFINE_list('image_size', [256, 256], 'image size to cv2 modle (w,h)')
tf.flags.DEFINE_integer('batch_size', 8, 'batch size')
tf.flags.DEFINE_integer('num_class', 61, 'class num')
tf.flags.DEFINE_integer('epochs', 200, 'class num')
tf.flags.DEFINE_integer('image_nums', 3120, 'image_nums')
tf.flags.DEFINE_list('transform_args',[random_transform_args], 'random transform args' )
tf.flags.DEFINE_boolean('use_tf_pipline', True, 'Is or not use tensorflow pipline with queue and mulit-thread')
print('FLAGS define complated.')
# print('absfds')
# return FLAGS



# from data.get_train_data import gen_train_data

# import data.get_train_data as get
# gen_train, val_train = get.gen_train_data(
#     train_json_path=tf.app.flags.FLAGS.train_json_path,
#     validate_json_path=tf.app.flags.FLAGS.validate_json_path,
#     train_data_dir=tf.app.flags.FLAGS.train_data_dir,
#     validate_data_dir=tf.app.flags.FLAGS.validate_data_dir,
#     image_size=tf.app.flags.FLAGS.image_size,
#     batch_size=tf.app.flags.FLAGS.batch_size,
#     trans_args=random_transform_args
#
# )
# g1 = gen_train()
# g2 = val_train()
# i = 0

import matplotlib.pyplot as plt

# x0,x1 = next(g1),next(g2)
# for x in g2:
#     i+=1
