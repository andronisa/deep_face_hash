# http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html
# Date - 05/07/2016

from __future__ import absolute_import
import numpy as np

from os.path import join, isdir
from os import listdir


# from sklearn.datasets import fetch_lfw_people


def load_lfw_db(data_fpath, n_batch=6):
    height = 250
    width = 250

    # for person_dir in sorted(listdir(data_fpath)):
    person_names, file_paths = [], []
    for person_name in sorted(listdir(data_fpath)):
        folder_path = join(data_fpath, person_name)
        if not isdir(folder_path):
            continue
        paths = [join(folder_path, f) for f in listdir(folder_path)]
        n_pictures = len(paths)
        person_name = person_name.replace('_', ' ')
        person_names.extend([person_name] * n_pictures)
        file_paths.extend(paths)

    num_faces = len(file_paths)

    target_names = np.unique(person_names)

    target = np.searchsorted(target_names, person_names)
    person_names = np.array(person_names)
    file_paths = np.array(file_paths)

    # shuffle the faces with a deterministic RNG scheme to avoid having
    # all faces of the same person in a row, as it would break some
    # cross validation and learning algorithms such as SGD and online
    # k-means that make an IID assumption

    indices = np.arange(num_faces)
    np.random.RandomState(42).shuffle(indices)
    file_paths, target, person_names = file_paths[indices], target[indices], person_names[indices]
    # print(file_paths[0])
    # print(target[0])
    # print(person_names[0])
    #
    # print(target.shape)
    # print(person_names.shape)
    # print(file_paths.shape)

    chunked_img_paths = np.array_split(file_paths, n_batch)
    chunked_targets = np.array_split(target, n_batch)
    chunked_names = np.array_split(person_names, n_batch)
    batch_size = chunked_img_paths[0].shape[0]

    img_options = {
        'height': height,
        'width': width,
        'num_faces': batch_size,
        'n_batch': n_batch
    }

    return chunked_img_paths, chunked_targets, chunked_names, img_options
    #
    #
    # def load_lfw_db():
    #     print("\nCalling 'fetch_lfw_people' ... \n")
    #     print("Brought them!")
    #
    #     return load
    #     #
    # # introspect the images arrays to find the shapes (for plotting)
    # n_samples, h, w, rgb = lfw_people.images.shape
    # first_img_dims = lfw_people.images[0].shape
    # print(first_img_dims)
    # exit()
    # # print(lfw_people.shape)
    # # print(lfw_people.images.shape)
    # # print(lfw_people.images[0])
    #
    # # for machine learning we use the 2 data directly (as relative pixel
    # # positions info is ignored by this model)
    # X = lfw_people.images
    #
    # n_features = lfw_people.data.shape[1]
    #
    # # the label to predict is the id of the person
    # y = lfw_people.target
    # target_names = lfw_people.target_names
    # n_classes = target_names.shape[0]
    #
    # print("Total dataset size:")
    # print("n_samples: %d" % n_samples)
    # print("n_features: %d" % n_features)
    # print("n_classes: %d" % n_classes)
    # print("n_target: %d" % len(np.unique(lfw_people['target'])))
    # print("Image Sizes: {0}x{1}".format(h, w))
    #
    # class_name_no_pairs = dict(zip(target_names, np.unique(lfw_people['target'])))
    # print(class_name_no_pairs)
    # #
    # # ###############################################################################
    # # # Split into a training set and a test set using a stratified k fold
    # #
    # # # split into a training, validation and testing sets
    # # x_train, X_test, Y_train, y_test = train_test_split(
    # #     X, y, test_size=0.10, random_state=42)
    # #
    # # X_train, X_valid, y_train, y_valid = train_test_split(
    # #     x_train, Y_train, test_size=0.25, random_state=42)
    # #
    # return lfw_people
    # # return (X_train, X_valid, X_test), (y_train, y_valid, y_test), (h, w, rgb, n_classes)


if __name__ == '__main__':
    lfw_people = load_lfw_db('/home/aandronis/scikit_learn_data/lfw_home/lfw/', color_mode='rgb')
