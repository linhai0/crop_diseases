import cv2
import numpy as np
import os, glob, sys, random
import tools.data_preprocess as datapre
import tools.utils as utils
import json
import tensorflow as tf


# import config
# from config import random_transform_args as trans_args


def _json2list(data_dir, json_path):
    with open(json_path, 'r') as f:
        json_file = json.load(f)

    if not json_file:
        raise FileNotFoundError(json_path)
    # {"disease_class": 1, "image_id": "62fd8bf4d53a1b94fbac16738406f10b.jpg"}

    L = [dict(image=os.path.join(data_dir, d['image_id']), label=d['disease_class']) for d in json_file]
    return L


def _load_image(batch_list, image_size=None, trans_args=None, *args, **kwargs):
    batch_x, batch_y = [], []
    for d in batch_list:
        img = cv2.imread(d['image'])
        if image_size:
            image_size = tuple(image_size)
            img = cv2.resize(img, image_size)
        ylabel = d['label']
        if trans_args:
            img = datapre.random_transform(image=img, **trans_args)
        img = datapre.normal_scalar01(img)
        batch_x.append(img)
        batch_y.append(ylabel)
    return np.stack(batch_x), np.stack(batch_y)


def _imbalance_dataset_split(train_list, validation_list):
    pass


def gen_train_data(FLAGS):
    train_json_path = FLAGS.train_json_path,
    validate_json_path = FLAGS.validate_json_path,
    train_data_dir = FLAGS.train_data_dir,
    validate_data_dir = FLAGS.validate_data_dir,
    image_size = FLAGS.image_size,
    batch_size = FLAGS.batch_size,
    trans_args = FLAGS.transform_args[0]
    use_imbalance = FLAGS.ues_imbalance

    train_list = _json2list(train_data_dir, train_json_path)
    validate_list = _json2list(validate_data_dir, validate_json_path)
    # train_all_label = [d['label'] for d in train_list]
    # # print(max(train_all_label), min(train_all_label))
    # print('class num: ', len({}.fromkeys(train_all_label)))
    # # utils.historm(train_all_label)

    if use_imbalance:
        train_list, validate_list = _imbalance_dataset_split(train_list, validate_list)

    print('Train data num: {0}\nValidate data num: {1}'.format(len(train_list), len(validate_list)))
    FLAGS.image_nums = len(train_list)

    def gen_train_batch():
        random.shuffle(train_list)

        for i in range(len(train_list) // batch_size):

            if (i + 1) * batch_size <= len(train_list):
                batch_list = train_list[batch_size * i:batch_size * (i + 1)]
            else:
                if i * batch_size == len(train_list):
                    continue
                else:
                    batch_list = train_list[batch_size * i:]
            yield _load_image(batch_list, image_size, trans_args)

    def gen_validate_batch():

        for i in range(len(validate_list) // batch_size):

            if (i + 1) * batch_size <= len(validate_list):
                batch_list = validate_list[batch_size * i:batch_size * (i + 1)]
            else:
                if i * batch_size == len(validate_list):
                    continue
                else:
                    batch_list = validate_list[batch_size * i:]
            yield _load_image(batch_list, image_size, trans_args=None)

    return gen_train_batch, gen_validate_batch


########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################


# def _tf_json2list(data_dir, json_path):
#     with open(json_path, 'r') as f:
#         json_file = json.load(f)
#
#     if not json_file:
#         raise FileNotFoundError(json_path)
#     # {"disease_class": 1, "image_id": "62fd8bf4d53a1b94fbac16738406f10b.jpg"}
#     nx, ny = [], []
#     for d in json_file:
#         nx.append(os.path.join(data_dir, d['image_id']))
#         ny.append(d['disease_class'])
#     return nx, ny  #[nx....], [ny....]


def tf_image_list_reader(FLAGS):
    train_json_path = FLAGS.train_json_path,
    validate_json_path = FLAGS.validate_json_path,
    train_data_dir = FLAGS.train_data_dir,
    validate_data_dir = FLAGS.validate_data_dir,
    image_size = FLAGS.image_size,
    batch_size = FLAGS.batch_size,
    trans_args = FLAGS.transform_args[0]
    use_imbalance = FLAGS.ues_imbalance
    train_list = _json2list(train_data_dir, train_json_path)
    validate_list = _json2list(validate_data_dir, validate_json_path)

    if use_imbalance:
        train_list, validate_list = _imbalance_dataset_split(train_list, validate_list)
    nx_train, nx_val, ny_train, ny_val = [], [], [], []
    for d in train_list:
        nx_train.append(d['image'])
        ny_train.append(d['label'])
    for d in validate_list:
        nx_val.append(d['image'])
        ny_val.append(d['label'])
    nx_train_tensor = tf.convert_to_tensor(nx_train)
    ny_train_tensor = tf.convert_to_tensor(ny_train)
    nx_val_tensor = tf.convert_to_tensor(nx_val)
    ny_val_tensor = tf.convert_to_tensor(ny_val)

    train_input_queue = tf.train.slice_input_producer([nx_train_tensor, ny_train_tensor], shuffle=True, capacity=32)
    val_input_queue = tf.train.slice_input_producer([nx_val_tensor, ny_val_tensor], shuffle=False, capacity=32)

    image_bytes = tf.read_file(train_input_queue[0])
    image = tf.read_file


def _data_list_to_tf_queue(data_list, batch_size, image_size, num_epochs=None, num_threads=1, capcity=32,
                           is_shuffle=True):
    """
    :param data_list: expected datalist = [images_list, labels], NEED: images format is png or jpeg format.
    :param batch_size:  bath size
    :param image_size:  image size
    :param num_epochs:  epoch_size, default=None mean to generation data forerver
    :param num_threads: threads default=1
    :param capcity: capcity, default=32
    :param is_shuffle:
    :return:  queue out as shape with [batch_images, batch_labels]
    """

    images_list_tensor = tf.conver_to_tensor(data_list[0])
    lables_list_tensor = tf.convert_to_tensor(data_list[1])
    input_queue = tf.train.slice_input_producer([images_list_tensor, lables_list_tensor], num_epochs=num_epochs,
                                                capcity=capcity, shuffle=is_shuffle)
    image_bytes = tf.convert_to_tensor(input_queue[0])
    image_de = tf.image.decode_jpeg(image_bytes, channels=3)  #
    image = _tf_agumete(image_de)
    batch_images, batch_labels = tf.train.batch([image, input_queue[1]], batch_size, num_threads=1)
    return batch_images, batch_labels


def _tf_agumete(image_queue, image_size, transform_args=None, num_channels=3, interpolation=0):
    """

    :param image_queue:
    :param image_size:
    :param transform_args:
    :param interpolation: 0,默认 双线性插值；1，最近邻算法； 2， 双3次插值法；3，面积插值法
    :return:
    """
    transform_args
