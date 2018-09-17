import cv2
import numpy as np
import tensorflow as tf


def random_transform(image, rotation_range, zoom_range, shift_range, random_flip):
    h, w = image.shape[0:2]
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    tx = np.random.uniform(-shift_range, shift_range) * w
    ty = np.random.uniform(-shift_range, shift_range) * h
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
    mat[:, 2] += (tx, ty)
    result = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if np.random.random() < random_flip:
        result = result[:, ::-1]
    return result


def normal_scalar01(images):
    return images / 255.


def spcial_precess(image):
    pass


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def crop(image, random_crop, image_size):
    if image.shape[1] > image_size:
        sz1 = int(image.shape[1] // 2)
        sz2 = int(image_size // 2)
        if random_crop:
            diff = sz1 - sz2
            (h, v) = (np.random.randint(-diff, diff + 1), np.random.randint(-diff, diff + 1))
        else:
            (h, v) = (0, 0)
        image = image[(sz1 - sz2 + v):(sz1 + sz2 + v), (sz1 - sz2 + h):(sz1 + sz2 + h), :]
    return image


def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image


def to_rgb(img):
    w, h = img.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = ret[:, :, 1] = ret[:, :, 2] = img
    return ret


class TfAugmentation(object):
    def __int__(self, image_queue, image_size, transform_args=None, num_channels=3, interpolation=0, bbox=None, *args,
                **kwargs):

        """
        :param image_queue:
        :param image_size:
        :param transform_args:
        :param interpolation: 0,默认 双线性插值；1，最近邻算法； 2， 双3次插值法；3，面积插值法
        :return:
        """
        self.image_queue = image_queue
        self.image_size = image_size
        self.trans_args = transform_args
        self.num_channels = num_channels
        self.interpolation = interpolation
        self.args_list = args
        self.kwargs_dict = kwargs
        self.bbox = bbox

    def distort_color(self, image, color_ordering=0):
        if color_ordering == 0:
            image = tf.image.random_brightness(image, max_delta=32. / 255.)  # 亮度
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  # 饱和度
            image = tf.image.random_hue(image, max_delta=0.2)  # 色相
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # 对比度
        elif color_ordering == 1:
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
        elif color_ordering == 2:
            image = tf.image.random_hue(image, max_delta=0.2)
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        elif color_ordering == 3:
            image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            image = tf.image.random_brightness(image, max_delta=32. / 255.)
            image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            image = tf.image.random_hue(image, max_delta=0.2)

        return tf.clip_by_value(image, .0, 1.)

    def use_bbox(self, image):
        bbox_begain, bbox_size = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=self.bbox)
        # tf.shape(image)进行queue的时候很容出错，慎用
        image = tf.slice(image, bbox_begain, bbox_size)
        return image

    def transform_image(self):
        image = tf.image.resize_images(self.image_queue, self.image_size, method=self.interpolation)



distorted_image = tf.image.resize_images(distorted_image, height, width, method=np.random.randint(4))
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = distort_color(distorted_image, np.random.randint(4))


distorted_image = tf.random_crop(image, size=[128, 64, num_channels])
    # distorted_image = tf.image.resize_images(image, (height, width), method=np.random.randint(4))
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_flip_up_down(distorted_image)
    distorted_image = distort_color(distorted_image, np.random.randint(4))
    distorted_image-tf.image.rot90(distorted_image,np.random.randint(4))


    fs ismf
fsgs
fs
getattr(sfs
        )fs

fsfs