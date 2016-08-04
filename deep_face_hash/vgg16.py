import os
import pickle
import cv2
import numpy as np

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


if __name__ == '__main__':
    if not os.path.isfile("test_feature_map.p"):
        im = cv2.resize(cv2.imread('test.jpg'), (224, 224))
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, axis=0)

        # Test pretrained model
        model = vgg_16(weights_path=WEIGHTS_PATH)
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')
        out = model.predict(im)

        print(out)
        print(type(out))
        print(out.shape)
        pickle.dump(out, open("test_feature_map.p", "wb"))
    else:
        feature_map = pickle.load(open("test_feature_map.p", "rb"))
        print(feature_map)


        # # Debug
        # print(model.summary())
