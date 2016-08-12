try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
from bson.binary import Binary
from keras.optimizers import SGD
from sklearn.metrics.pairwise import pairwise_distances

from data.img_utils import preprocess_images
from data.lfw_db import load_lfw_db
from data.storage import mongodb_store, mongodb_find, clear_collection
from bit_partition_lsh import generate_hash_maps, generate_hash_vars
from vgg16 import generate_feature_maps, WEIGHTS_PATH, vgg_16, get_feature_no


def deep_face_hash(window, hash_size=64, reset_db=False, existing_maps=False):
    print("\n##################### INITIALIZE #########################")

    height = 250
    width = 250
    window = window if window else 1600
    print(window)

    print("Loading model...")
    # Test pretrained model
    model = vgg_16(weights_path=WEIGHTS_PATH, h=height, w=width)
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    (chunked_img_paths, chunked_targets, chunked_names, img_options) = load_lfw_db(
        '/home/aandronis/scikit_learn_data/lfw_home/lfw/')

    print("\nStarting Hashing...")
    print("Total batches: " + str(img_options['n_batch']))
    print("Images per batch: " + str(img_options['num_faces']))

    img_size = None
    crop_size = None
    color_mode = "rgb"

    if reset_db:
        print("\n##################### DATABASE #########################")
        clear_collection(hash_size)
        clear_collection()

    # compute one feature map to get dimensionality
    img = preprocess_images([chunked_img_paths[0][0]])[0]
    feature_no = get_feature_no(img, model)
    hash_vars = generate_hash_vars(dim_size=feature_no, window_size=window, bits=hash_size)

    batch_counter = 0
    for img_paths in chunked_img_paths:
        preprocessed_images = None
        print("\n#################### CHUNK HANDLING ##########################")
        print("Starting image batch no." + str(batch_counter + 1) + "\n")

        if not existing_maps:
            print("Preprocessing Images...")
            preprocessed_images = preprocess_images(img_paths.tolist(), img_size, crop_size, color_mode, img_options)
            print("Generating Feature Maps...")
            feature_maps = generate_feature_maps(preprocessed_images, model)
        else:
            print("Using Preprocessed Feature Maps...")
            q = {}
            f = {'feature_map': 1}
            feature_maps = map(pickle.loads, [item['feature_map'] for item in mongodb_find(q, f)])

        hash_codes = generate_hash_maps(feature_maps, hash_vars)

        def convert_to_binary(feat_map):
            return Binary(pickle.dumps(feat_map, protocol=2))

        feature_map_binaries = map(convert_to_binary, feature_maps)

        if not existing_maps:
            batch_list = zip(feature_map_binaries, hash_codes, chunked_targets[batch_counter].tolist(),
                             chunked_names[batch_counter].tolist())
        else:
            batch_list = zip(feature_map_binaries, hash_codes, chunked_targets,
                             chunked_names)

        mongodb_store(batch_list)

        del preprocessed_images
        del feature_maps
        del hash_codes
        del feature_map_binaries
        del batch_list

        print("\n##############################################")
        print("Finished image batch no." + str(batch_counter + 1))
        print("##############################################\n")
        # print(hash_codes)

        batch_counter += 1
    return


def calculate_mean_window_size():
    q = {}
    f = {'feature_map': 1}
    l = 1000

    try:
        feature_maps = np.array(map(pickle.loads, [item['feature_map'] for item in mongodb_find(q, f, l)]))
        feat_maps = feature_maps.reshape(feature_maps.shape[0], feature_maps.shape[2])
        distances = pairwise_distances(feat_maps)
    except Exception as ex:
        return None

    return np.mean(distances)


if __name__ == '__main__':
    mean_window_size = calculate_mean_window_size()
    deep_face_hash(mean_window_size, None, reset_db=True, existing_maps=False)
