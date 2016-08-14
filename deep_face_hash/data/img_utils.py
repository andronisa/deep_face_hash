import cv2
import numpy as np

from scipy.misc import imread


def preprocess_images(image_paths, img_size=None, crop_size=None, color_mode="rgb", img_options=None):
    if img_options:
        height = img_options['height']
        width = img_options['width']
        num_faces = img_options['num_faces']
        rgb = 3 if color_mode == 'rgb' else None
    else:
        height = 250
        width = 250
        num_faces = 1
        rgb = 3

    images = np.zeros((num_faces, rgb, height, width), dtype=np.float32)

    counter = 0
    for image_path in image_paths:
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

        images[counter, ...] = img
        counter += 1

    return images
