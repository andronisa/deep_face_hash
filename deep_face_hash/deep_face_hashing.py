try:
    import cPickle as pickle
except ImportError:
    import pickle

from bson.binary import Binary
from keras.optimizers import SGD

from data.img_utils import preprocess_images
from data.lfw_db import load_lfw_db
from data.storage import store_to_mongo, clear_collection
from bit_partition_lsh import generate_hash_maps, generate_hash_vars
from vgg16 import generate_feature_maps, WEIGHTS_PATH, vgg_16, get_feature_no


def test_hashing_lfw(window, hash_size):
    print("\n##################### INITIALIZE #########################")

    height = 250
    width = 250

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

    print("\n##################### DATABASE #########################")
    print("Clear Mongo Collection...")
    clear_collection()

    # compute one feature map to get dimensionality
    img = preprocess_images([chunked_img_paths[0][0]])[0]
    feature_no = get_feature_no(img, model)
    hash_vars = generate_hash_vars(dim_size=feature_no, window_size=window, bits=hash_size)

    batch_counter = 0
    for img_paths in chunked_img_paths:
        print("\n#################### CHUNK HANDLING ##########################")
        print("Starting image batch no." + str(batch_counter+1) + "\n")
        preprocessed_images = preprocess_images(img_paths.tolist(), img_size, crop_size, color_mode, img_options)

        feature_maps = generate_feature_maps(preprocessed_images, model)
        hash_codes = generate_hash_maps(feature_maps, hash_vars)

        def convert_to_binary(feat_map):
            return Binary(pickle.dumps(feat_map, protocol=2))

        feature_map_binaries = map(convert_to_binary, feature_maps)
        batch_list = zip(feature_map_binaries, hash_codes, chunked_targets[batch_counter].tolist(),
                         chunked_names[batch_counter].tolist())
        store_to_mongo(batch_list)

        del(preprocessed_images)
        del(feature_maps)
        del(hash_codes)
        del(feature_map_binaries)
        del(batch_list)

        print("\n##############################################")
        print("Finished image batch no." + str(batch_counter+1))
        print("##############################################\n")
        # print(hash_codes)

        batch_counter += 1
    return


if __name__ == '__main__':
    test_hashing_lfw(500, 64)