import os
import cv2
import pickle
import editdistance
import numpy as np

from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD

from models import DeepFaceHashSequential
from alternative_hashing import dhash
from data.img_utils import preprocess_images
from utils import hamming_distance
from data.storage import mongodb_store, clear_collection
from data.lfw_db import load_lfw_db
from utils import arr_to_binary

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


def load_model():
    print("Loading model...")

    height = 224
    width = 224

    model = vgg_16(weights_path=WEIGHTS_PATH, h=height, w=width)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    return model


def get_feature_no(img, model):
    img = np.expand_dims(img, axis=0)
    feature_map = model.predict(img)

    return feature_map.shape[1]


def generate_feature_maps(images, model, names, insert=False):
    print("\nGenerating Feature Maps...")
    feature_maps = []
    counter = 0

    for img in images:
        img = np.expand_dims(img, axis=0)
        feature_map = model.predict(img)
        feature_maps.append(feature_map)
        del feature_map

        counter += 1
        if counter % 500 == 0:
            print("Generated " + str(counter) + " feature maps")

    print("Generated " + str(counter) + " total feature maps")

    if insert:
        # Storing to mongo
        mongodb_store(zip(map(arr_to_binary, feature_maps), names.tolist()), keys=['feature_map', 'name'], collection='feature_maps_final')

    return feature_maps


def test_hashing():
    height = 224
    width = 224

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

    print(hamming_distance(hash_code, hash_code_2))
    print(hamming_distance(hash_code, hash_code_3))
    print(hamming_distance(hash_code, hash_code_4))

    # print(hamming_distance(hash_code_2, hash_code))
    # print(hamming_distance(hash_code_2, hash_code_3))
    # print(hamming_distance(hash_code_2, hash_code_4))

    # print("\nLEVENSTEIN\n")
    # print(editdistance.eval(hash_code_3, hash_code))
    # print(editdistance.eval(hash_code_3, hash_code_2))
    # print(editdistance.eval(hash_code_3, hash_code_4))
    #
    # print(editdistance.eval(hash_code_4, hash_code))
    # print(editdistance.eval(hash_code_4, hash_code_2))


def test_feature_map_generation_and_storage():
    # clear_collection('feature_maps')
    print("Testing feat map generation and storage...")
    (chunked_img_paths, chunked_targets, chunked_names, img_options) = load_lfw_db(
        data_fpath='/home/aandronis/scikit_learn_data/lfw_home/lfw/')
    lfw_model = load_model()

    batch_counter = 0
    for img_paths in chunked_img_paths:
        print("Starting image batch no." + str(batch_counter + 1) + "\n")
        print("Preprocessing Images...")

        names = chunked_names[batch_counter]
        preprocessed_images = preprocess_images(img_paths.tolist(), img_size=(224, 224), img_options=img_options)
        feature_maps = generate_feature_maps(preprocessed_images, lfw_model, names, insert=True)

        del preprocessed_images
        del feature_maps

        batch_counter += 1


if __name__ == '__main__':
    print("Choose a function to act")
    # test_hashing()
    test_feature_map_generation_and_storage()
