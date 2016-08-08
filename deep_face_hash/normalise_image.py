import cv2
import numpy as np

from scipy.misc import imread


def preprocess_image(image_path, img_size=None, crop_size=None, color_mode="rgb", out=None):
    img = imread(image_path, mode='RGB')
    if img_size:
        img = cv2.resize(img, img_size)

    img = img.astype('float32')
    # We permute the colors to get them in the BGR order
    if color_mode == "bgr":
        img[:, :, [0, 1, 2]] = img[:, :, [2, 1, 0]]
    # We normalize the colors with the empirical means on the training set
    img[:, :, 0] -= 123.68
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 103.939
    img = img.transpose((2, 0, 1))

    if crop_size:
        img = img[:, (img_size[0] - crop_size[0]) // 2:(img_size[0] + crop_size[0]) // 2
        , (img_size[1] - crop_size[1]) // 2:(img_size[1] + crop_size[1]) // 2]

    img = np.expand_dims(img, axis=0)

    return img
