import os
import pickle
import cv2
import numpy as np
import editdistance

from normalise_image import preprocess_image
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

from models import DeepFaceHashSequential

WEIGHTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "weights", "vgg16_weights.h5"))


def vgg_16(weights_path=None):
    model = DeepFaceHashSequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten(name="flatten"))

    # model.add(Dense(4096, activation='relu', name='dense_1'))
    # model.add(Dropout(0.5))

    # model.add(Dense(4096, activation='relu', name='dense_2'))
    # model.add(Dropout(0.5))

    # model.add(Dense(1000, name='dense_3'))
    # model.add(Activation("softmax",name="softmax"))

    if weights_path:
        print("Trying to load weights...")

        excluded = ['dense_1', 'dense_2', 'dense_3', 'dropout_5', 'dropout_6' 'softmax']
        model.load_weights(weights_path, excluded)

        print("Weights loaded!!!")
    return model


from PIL import Image
import numpy


# import scipy.fftpack
# import pywt

def _binary_array_to_hex(arr):
    """
    internal function to make a hex string out of a binary array
    """
    h = 0
    s = []
    for i, v in enumerate(arr.flatten()):
        if v:
            h += 2 ** (i % 8)
        if (i % 8) == 7:
            s.append(hex(h)[2:].rjust(2, '0'))
            h = 0
    return "".join(s)


class ImageHash(object):
    """
    Hash encapsulation. Can be used for dictionary keys and comparisons.
    """

    def __init__(self, binary_array):
        self.hash = binary_array

    def __str__(self):
        return _binary_array_to_hex(self.hash.flatten())

    def __repr__(self):
        return repr(self.hash)

    def __sub__(self, other):
        if other is None:
            raise TypeError('Other hash must not be None.')
        if self.hash.size != other.hash.size:
            raise TypeError('ImageHashes must be of the same shape.', self.hash.shape, other.hash.shape)
        return (self.hash.flatten() != other.hash.flatten()).sum()

    def __eq__(self, other):
        if other is None:
            return False
        return numpy.array_equal(self.hash.flatten(), other.hash.flatten())

    def __ne__(self, other):
        if other is None:
            return False
        return not numpy.array_equal(self.hash.flatten(), other.hash.flatten())

    def __hash__(self):
        # this returns a 8 bit integer, intentionally shortening the information
        return sum([2 ** (i % 8) for i, v in enumerate(self.hash.flatten()) if v])


def hex_to_hash(hexstr):
    """
    Convert a stored hash (hex, as retrieved from str(Imagehash))
    back to a Imagehash object.
    """
    l = []
    if len(hexstr) != 16:
        raise ValueError('The hex string has the wrong length')
    for i in range(8):
        h = hexstr[i * 2:i * 2 + 2]
        v = int("0x" + h, 16)
        l.append([v & 2 ** i > 0 for i in range(8)])
    return ImageHash(numpy.array(l))


if __name__ == '__main__':
    # im = cv2.resize(cv2.imread('data/img/tony_1.jpg'), (224, 224))
    # im = im.transpose((2, 0, 1))
    # im = np.expand_dims(im, axis=0)

    im = preprocess_image('data/img/tony_1.jpg', img_size=(224, 224), color_mode='rgb')
    im2 = preprocess_image('data/img/tony_2.jpg', img_size=(224, 224), color_mode='rgb')
    im3 = preprocess_image('data/img/tony_3.jpg', img_size=(224, 224), color_mode='rgb')
    im4 = preprocess_image('data/img/nicola_3.jpg', img_size=(224, 224), color_mode='rgb')

    # Test pretrained model
    model = vgg_16(weights_path=WEIGHTS_PATH)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    feature_map = model.predict(im)
    feature_map_2 = model.predict(im2)
    feature_map_3 = model.predict(im3)
    feature_map_4 = model.predict(im4)

    #
    # get_feature = theano.function([model.layers[0].input], model.layers[30].output,
    #                               allow_input_downcast=False)
    # feat = get_feature(im)
    # print(feat.shape)
    # plt.imshow(feat[0][511])
    # plt.show()

    # print(feat_map)
    # print(type(out))
    # print(out.shape)
    # pickle.dump(out, open("test_feature_map.p", "wb"))

    feat_map = feature_map.transpose().flatten()
    new_ft_map = np.append(feat_map, np.zeros(34))
    pixels = new_ft_map.reshape((158 + 1, 158))
    # compute differences
    diff = pixels[1:, :] > pixels[:-1, :]
    hash_code = ImageHash(diff).__str__()
    # print(hash_code.__str__())
    # exit()

    feat_map_2 = feature_map_2.transpose().flatten()
    new_ft_map_2 = np.append(feat_map_2, np.zeros(34))
    pixels_2 = new_ft_map_2.reshape((158 + 1, 158))
    # compute differences
    diff_2 = pixels_2[1:, :] > pixels_2[:-1, :]
    hash_code_2 = ImageHash(diff_2).__str__()

    feat_map_3 = feature_map_3.transpose().flatten()
    new_ft_map_3 = np.append(feat_map_3, np.zeros(34))
    pixels_3 = new_ft_map_3.reshape((158 + 1, 158))
    # compute differences
    diff_3 = pixels_3[1:, :] > pixels_3[:-1, :]
    hash_code_3 = ImageHash(diff_3).__str__()

    feat_map_4 = feature_map_4.transpose().flatten()
    new_ft_map_4 = np.append(feat_map_4, np.zeros(34))
    pixels_4 = new_ft_map_4.reshape((158 + 1, 158))
    # compute differences
    diff_4 = pixels_4[1:, :] > pixels_4[:-1, :]
    hash_code_4 = ImageHash(diff_4).__str__()


    # print(hash_code_2)
    def hamming2(s1, s2):
        """Calculate the Hamming distance between two bit strings"""
        assert len(s1) == len(s2)
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))


    #
    # print(hamming2(hash_code, hash_code_2))
    # print(hamming2(hash_code, hash_code_3))
    # print(hamming2(hash_code, hash_code_4))
    #
    # print(hamming2(hash_code_2, hash_code_3))
    # print(hamming2(hash_code_2, hash_code_3))
    # print(hamming2(hash_code_2, hash_code_3))

    print(hamming2(hash_code, hash_code))
    print(hamming2(hash_code, hash_code_2))
    print(hamming2(hash_code, hash_code_3))
    print(hamming2(hash_code, hash_code_4))

    # print(hamming2(hash_code_2, hash_code))
    # print(hamming2(hash_code_2, hash_code_3))
    # print(hamming2(hash_code_2, hash_code_4))

    # print("\nLEVENSTEIN\n")
    # print(editdistance.eval(hash_code_3, hash_code))
    # print(editdistance.eval(hash_code_3, hash_code_2))
    # print(editdistance.eval(hash_code_3, hash_code_4))
    #
    # print(editdistance.eval(hash_code_4, hash_code))
    # print(editdistance.eval(hash_code_4, hash_code_2))

    # print(hamming(hash_code, hash_code_2))
    # print(hamming(hash_code, hash_code_3))
    # print(hamming(hash_code_2, hash_code_3))
