import os, json, time, random, sys
import tensorflow as tf

# I personally always like to make my paths absolute
# to be independent from where the python binary is called
dir = os.path.dirname(os.path.realpath(__file__))
dataset_dir = '/home/linhai/Workspace/crop_diseases/data'
random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.4,

    'horizontal_flip': False,
    'vertical_flip': False,
    'corp_probility': 0,
    'crop_min_percent': .6,
    'crop_max_percent': 1.,
    'mixup': 4,
}


# I won't dig into TF interaction with the shell, feel free to explore the documentation

def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()
    keys_list = [keys for keys in flags_dict]
    for keys in keys_list:
        FLAGS.__delattr__(keys)


# def get_flags():
del_all_flags(tf.flags.FLAGS)
FLAGS = tf.flags.FLAGS
# tf.summary.histogram
# tf.flags
# Hyper-parameters search configuration
tf.flags.DEFINE_boolean('fullsearch', False,
                        'Perform a full search of hyperparameter space ex:(hyperband -> lr search -> hyperband with best lr)')

# Agent configuration
tf.flags.DEFINE_string('model_name', 'sim_net', 'Unique name of the model')
tf.flags.DEFINE_list('image_size', [256, 256], 'Image size to cv2 modle (w,h)')
tf.flags.DEFINE_integer('batch_size', 32, 'Batch size')
tf.flags.DEFINE_integer('num_classes', 61, 'Class num')
tf.flags.DEFINE_integer('num_epochs', 200, 'Class num')
tf.flags.DEFINE_list('transform_args', [random_transform_args], 'Random transform args')
tf.flags.DEFINE_boolean('use_tf_pipline', True, 'Is or not use tensorflow pipline with queue and mulit-thread')
tf.flags.DEFINE_integer('num_channels', 3, 'Images channels ')
tf.flags.DEFINE_boolean('is_loadmodel', True, 'Is or not restore traned model last time.')

# tf.flags.DEFINE_boolean('best', False, 'Force to use the best known configuration')
# tf.flags.DEFINE_float('initial_mean', 0., 'Initial mean for NN')
# tf.flags.DEFINE_float('initial_stddev', 1e-3, 'Initial standard deviation for NN')
# Train parameter
tf.flags.DEFINE_float('learning_rate', 1e-3, 'The learning rate of SGD/ADM')
tf.flags.DEFINE_float('decay_rate','.9', 'learning rate decay rate')
tf.flags.DEFINE_float('moving_average_decay', '.9', 'moving_average_decay')
tf.flags.DEFINE_boolean('log_histograms', True, 'whether use histograms log')
# tf.flags.DEFINE_float('nb_units', 20, 'Number of hidden units in Deep learning agents')

# # Environment configuration
# tf.flags.DEFINE_boolean('debug', False, 'Debug mode')
tf.flags.DEFINE_boolean('infer', False, 'Load an agent for playing')

tf.flags.DEFINE_boolean('use_imbalance', False, 'Whether use imbalance to split dataset include train and validate')

# This is very important for TensorBoard
# each model will end up in its own unique folder using time module
# Obviously one can also choose to name the output folder
tf.flags.DEFINE_string('output_dir', dir + '/output/' + FLAGS.model_name + '/',  # + str(int(time.time())),
                       'Name of the directory to store/log the model (if it exists, the model will be loaded from it)')
tf.flags.DEFINE_string('log_dir', dir + '/output/log_dir/', 'Log dir')
tf.flags.DEFINE_float('log_interval', '100', 'log per-steps')

# Another important point, you must provide an access to the random seed
# to be able to fully reproduce an experiment
tf.flags.DEFINE_integer('random_seed', random.randint(0, sys.maxsize), 'Value of random seed')

tf.flags.DEFINE_string('train_json_path',
                       dataset_dir + '/AgriculturalDisease_trainingset/AgriculturalDisease_train_annotations.json',
                       'Train dataset json path')
tf.flags.DEFINE_string('validate_json_path',
                       dataset_dir + '/AgriculturalDisease_validationset/AgriculturalDisease_validation_annotations.json',
                       'Validate dataset json path')
tf.flags.DEFINE_string('train_data_dir',
                       dataset_dir + '/AgriculturalDisease_trainingset/images',
                       'Train data dir')
tf.flags.DEFINE_string('validate_data_dir',
                       dataset_dir + '/AgriculturalDisease_validationset/images',
                       'Validate data dir')

print('FLAGS define complated.')

"""
添加优化器类型
"""
