import cv2
import numpy as np
import os, glob, sys, random
import tools.data_preprocess as datapre
from tools.data_preprocess import TfAugmentation
import tools.utils as utils
import json
import tensorflow as tf


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


########################################################################################################################
########################################################################################################################
# generator image reader
########################################################################################################################
########################################################################################################################
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
# tf queue image reader with pipline
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


def _data_list_to_tf_queue(data_list, batch_size, image_size, transform_args, num_channels=3, num_epochs=None,
                           num_threads=1, capcity=32,
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

    images_list_tensor = tf.convert_to_tensor(data_list[0])
    lables_list_tensor = tf.convert_to_tensor(data_list[1])
    input_queue = tf.train.slice_input_producer([images_list_tensor, lables_list_tensor], num_epochs=num_epochs,
                                                capcity=capcity, shuffle=is_shuffle)
    image_bytes = tf.convert_to_tensor(input_queue[0])
    image_de = tf.image.decode_jpeg(image_bytes, channels=3)  #
    resize_image = tf.image.resize_images(image_de, image_size)

    batch_images, batch_labels = tf.train.batch([resize_image, input_queue[1]], batch_size, num_threads=num_threads)

    aug = TfAugmentation(batch_size=batch_size, image_size=image_size, transform_args=transform_args)

    b_images, b_labels = aug.augment(batch_images, batch_labels)

    return b_images, b_labels


def tf_image_list_reader(FLAGS):
    train_json_path = FLAGS.train_json_path,
    validate_json_path = FLAGS.validate_json_path,
    train_data_dir = FLAGS.train_data_dir,
    validate_data_dir = FLAGS.validate_data_dir,
    image_size = FLAGS.image_size,
    batch_size = FLAGS.batch_size,
    trans_args = FLAGS.transform_args[0]
    num_epochs = FLAGS.num_epochs
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
    with tf.device('/cpu:0'):  # use cpu
        nx_train_tensor = tf.convert_to_tensor(nx_train)
        ny_train_tensor = tf.convert_to_tensor(ny_train)
        nx_val_tensor = tf.convert_to_tensor(nx_val)
        ny_val_tensor = tf.convert_to_tensor(ny_val)

        train_input_queue = tf.train.slice_input_producer([nx_train_tensor, ny_train_tensor], shuffle=True, capacity=32)
        val_input_queue = tf.train.slice_input_producer([nx_val_tensor, ny_val_tensor], shuffle=False, capacity=32)

        bx_train, by_train = _data_list_to_tf_queue(train_input_queue, batch_size, image_size, trans_args)
        bx_val, by_val = _data_list_to_tf_queue(val_input_queue, batch_size, image_size, trans_args)

    return bx_train, by_train, bx_val, by_val


########################################################################################################################
########################################################################################################################
# muti-process image reader with cpu
########################################################################################################################
########################################################################################################################
def multi_process_data_read(FLAGS):
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

    print('Train data num: {0}\nValidate data num: {1}'.format(len(train_list), len(validate_list)))
    FLAGS.image_nums = len(train_list)
    train_batchs = len(train_list) // FLAGS.batch_size
    has_smaller_batch = 0 if validate_list % batch_size == 0 else 1
    valid_batchs = len(validate_list) // FLAGS.batch_size + has_smaller_batch

    def train_batch_func(q1):
        random.shuffle(train_list)
        i = 0
        while True:
            # if i >
            tmp_list = train_list[i * batch_size:(i + 1) * batch_size]
            i += 1
            if i == train_batchs:
                i = 0
                random.shuffle(train_list)

            q1.put((_load_image(tmp_list, image_size, trans_args)))

    def valid_batch_func(q2):

        i = 0
        if has_smaller_batch:
            while True:
                if i == valid_batchs - 1:
                    tmp_list = validate_list[i * batch_size:]
                    i = 0
                else:
                    tmp_list = validate_list[i * batch_size:(i + 1) * batch_size]

                i += 1
                q2.put(_load_image(tmp_list, image_size, trans_args))
        else:
            while True:
                tmp_list = validate_list[i * batch_size:(i + 1) * batch_size]
                i += 1
                if i == valid_batchs:
                    i = 0
                # multi-process
                q2.put((_load_image(tmp_list, image_size, trans_args)))

    return train_batch_func, valid_batch_func, [train_batchs, valid_batchs]
