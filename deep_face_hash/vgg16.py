import os
import pickle
import editdistance
import numpy as np

from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

from models import DeepFaceHashSequential
from alternative_hashing import dhash
from data.img_utils import preprocess_images

WEIGHTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "weights", "vgg16_weights.h5"))


def vgg_16(weights_path=None, h=224, w=224):
    model = DeepFaceHashSequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(3, h, w)))
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


def get_feature_no(img, model):
    img = np.expand_dims(img, axis=0)
    feature_map = model.predict(img)

    return feature_map.shape[1]


def generate_feature_maps(images, model):
    feature_maps = []
    counter = 0
    for img in images:
        img = np.expand_dims(img, axis=0)
        feature_map = model.predict(img)
        feature_maps.append(feature_map)
        counter += 1

        if counter % 500 == 0:
            print("Generated " + str(counter) + " feature maps")

            # if counter % 10 == 0:
            #     return feature_maps

    print("Generated " + str(counter) + " total feature maps")
    return feature_maps


def test_hashing():
    height = 250
    width = 250

    im = preprocess_images(['data/img/tony_1.jpg'], img_size=(height, width), color_mode='rgb')
    im2 = preprocess_images(['data/img/tony_2.jpg'], img_size=(height, width), color_mode='rgb')
    im3 = preprocess_images(['data/img/tony_3.jpg'], img_size=(height, width), color_mode='rgb')
    im4 = preprocess_images(['data/img/nicola_3.jpg'], img_size=(height, width), color_mode='rgb')
    im5 = preprocess_images(['data/img/elina_1.jpg'], img_size=(height, width), color_mode='rgb')
    im6 = preprocess_images(['data/img/elina_2.jpg'], img_size=(height, width), color_mode='rgb')
    im7 = preprocess_images(['data/img/nikos.jpg'], img_size=(height, width), color_mode='rgb')
    im8 = preprocess_images(['data/img/test.jpg'], img_size=(height, width), color_mode='rgb')

    # Test pretrained model
    model = vgg_16(weights_path=WEIGHTS_PATH, h=height, w=width)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    feature_map = model.predict(im)
    feature_map_2 = model.predict(im2)
    feature_map_3 = model.predict(im3)
    feature_map_4 = model.predict(im4)
    feature_map_5 = model.predict(im5)
    feature_map_6 = model.predict(im6)
    feature_map_7 = model.predict(im7)
    feature_map_8 = model.predict(im8)

    pickle.dump(feature_map, open("data/feature_map.p", "wb"))
    pickle.dump(feature_map_2, open("data/feature_map_2.p", "wb"))
    pickle.dump(feature_map_3, open("data/feature_map_3.p", "wb"))
    pickle.dump(feature_map_4, open("data/feature_map_4.p", "wb"))
    pickle.dump(feature_map_5, open("data/feature_map_5.p", "wb"))
    pickle.dump(feature_map_6, open("data/feature_map_6.p", "wb"))
    pickle.dump(feature_map_7, open("data/feature_map_7.p", "wb"))
    pickle.dump(feature_map_8, open("data/feature_map_8.p", "wb"))

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

    hash_code = dhash(feature_map, hash_size=158)
    # print(hash_code.__str__())

    hash_code_2 = dhash(feature_map_2, hash_size=158)
    hash_code_3 = dhash(feature_map_3, hash_size=158)
    hash_code_4 = dhash(feature_map_4, hash_size=158)

    # hash_code_5 = dhash(feature_map_5, hash_size=158)
    # hash_code_6 = dhash(feature_map_6, hash_size=158)
    # hash_code_7 = dhash(feature_map_7, hash_size=158)
    # hash_code_8 = dhash(feature_map_8, hash_size=158)

    # print(hash_code_2)
    def hamming(s1, s2):
        """Calculate the Hamming distance between two bit strings"""
        assert len(s1) == len(s2)
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    print(hamming(hash_code, hash_code_2))
    print(hamming(hash_code, hash_code_3))
    print(hamming(hash_code, hash_code_4))

    # print(hamming(hash_code_2, hash_code))
    # print(hamming(hash_code_2, hash_code_3))
    # print(hamming(hash_code_2, hash_code_4))

    # print("\nLEVENSTEIN\n")
    # print(editdistance.eval(hash_code_3, hash_code))
    # print(editdistance.eval(hash_code_3, hash_code_2))
    # print(editdistance.eval(hash_code_3, hash_code_4))
    #
    # print(editdistance.eval(hash_code_4, hash_code))
    # print(editdistance.eval(hash_code_4, hash_code_2))


if __name__ == '__main__':
    test_hashing()
