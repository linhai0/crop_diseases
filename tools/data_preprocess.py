import cv2
import numpy as np
import tensorflow as tf
import math


def random_transform(image, rotation_range, zoom_range, shift_range, random_flip, *args, **kwargs):
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
    def __init__(self, batch_size, image_size, transform_args=None, num_channels=3, interpolation=0, bbox=None, *args,
                **kwargs):

        """
        :param image_queue:
        :param image_size:
        :param transform_args:
        :param interpolation: 0,默认 双线性插值；1，最近邻算法； 2， 双3次插值法；3，面积插值法
        :return:
        """
        self.batch_size = batch_size
        self.image_size = image_size
        self.trans_args = transform_args
        self.num_channels = num_channels
        self.interpolation = interpolation
        self.bbox = bbox
        self.args_list = args
        self.kwargs_dict = kwargs

    def distort_color(self, image, color_ordering=0):
        # single image not images
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
        # single images
        bbox_begain, bbox_size = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=self.bbox)
        # tf.shape(image)进行queue的时候很容出错，慎用
        image = tf.slice(image, bbox_begain, bbox_size)
        return image

    def augment(self, images, labels):
        return self._inter_augment(images, labels, self.batch_size, self.image_size, **self.trans_args)

    # GPU Aumentation
    def _inter_augment(self, images, labels, batch_size, image_size,
                       horizontal_flip=True,
                       vertical_flip=True,
                       rotation_range=8,  # Maximum rotation angle in degrees
                       crop_probability=0.8,  # How often we do crops
                       crop_min_percent=0.6,  # Minimum linear dimension of a crop
                       crop_max_percent=1.,  # Maximum linear dimension of a crop
                       mixup=0):  # Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf

        # My experiments showed that casting on GPU improves training performance
        labels = tf.to_float(labels)

        with tf.name_scope('augmentation'):
            width = tf.cast(image_size[1], tf.float32)
            height = tf.cast(image_size[0], tf.float32)

            # The list of affine transformations that our image will go under.
            # Every element is Nx8 tensor, where N is a batch size.
            transforms = []
            identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
            if horizontal_flip:
                coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
                flip_transform = tf.convert_to_tensor(
                    [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
                transforms.append(
                    tf.where(coin,
                             tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                             tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

            if vertical_flip:
                coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
                flip_transform = tf.convert_to_tensor(
                    [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
                transforms.append(
                    tf.where(coin,
                             tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                             tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

            if rotation_range > 0:
                angle_rad = rotation_range / 180 * math.pi
                angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
                transforms.append(
                    tf.contrib.image.angles_to_projective_transforms(
                        angles, height, width))

            if crop_probability > 0:
                crop_pct = tf.random_uniform([batch_size], crop_min_percent,
                                             crop_max_percent)
                left = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
                top = tf.random_uniform([batch_size], 0, height * (1 - crop_pct))
                crop_transform = tf.stack([
                    crop_pct,
                    tf.zeros([batch_size]), top,
                    tf.zeros([batch_size]), crop_pct, left,
                    tf.zeros([batch_size]),
                    tf.zeros([batch_size])
                ], 1)

                coin = tf.less(
                    tf.random_uniform([batch_size], 0, 1.0), crop_probability)
                transforms.append(
                    tf.where(coin, crop_transform,
                             tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

            if transforms:
                images = tf.contrib.image.transform(
                    images,
                    tf.contrib.image.compose_transforms(*transforms),
                    interpolation='BILINEAR')  # or 'NEAREST'

            def cshift(values):  # Circular shift in batch dimension
                return tf.concat([values[-1:, ...], values[:-1, ...]], 0)

            if mixup > 0:
                mixup = 1.0 * mixup  # Convert to float, as tf.distributions.Beta requires floats.
                beta = tf.distributions.Beta(mixup, mixup)
                lam = beta.sample(batch_size)
                ll = tf.expand_dims(tf.expand_dims(tf.expand_dims(lam, -1), -1), -1)
                images = ll * images + (1 - ll) * cshift(images)
                labels = lam * labels + (1 - lam) * cshift(labels)

        return images, labels

    def mix_not_use(self, image, distorted_image, height, width):
        distorted_image = tf.image.resize_images(distorted_image, height, width, method=np.random.randint(4))
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        distorted_image = self.distort_color(distorted_image, np.random.randint(4))

        distorted_image = tf.random_crop(image, size=[128, 64, self.num_channels])
        # distorted_image = tf.image.resize_images(image, (height, width), method=np.random.randint(4))
        distorted_image = tf.image.random_flip_left_right(distorted_image)
        distorted_image = tf.image.random_flip_up_down(distorted_image)
        distorted_image = self.distort_color(distorted_image, np.random.randint(4))
        distorted_image - tf.image.rot90(distorted_image, np.random.randint(4))

        # 随机裁剪，随机左右翻转，颜色失真

        def transform_image(self):
            image = tf.image.resize_images(self.image_queue, self.image_size, method=self.interpolation)
            # resize_images must use befor tf.train.batch()

            # 单张图片
            tf.random_crop()

            tf.image.per_image_standardization()

            # batch图片
            # image = tf.image.rot90(image, np.random.randint(0, 4))
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)

        def central_scale_images(self, image, scales):
            pass  # boxes

