# http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html
# Date - 05/07/2016

from __future__ import absolute_import
import numpy as np

from os.path import join, isdir
from os import listdir, walk


# from sklearn.datasets import fetch_lfw_people


def load_lfw_db(data_fpath='/home/aandronis/scikit_learn_data/lfw_home/lfw/', n_batch=6):
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


def load_images(data_fpath):
    paths = []
    for root, dirs, file_names in walk(data_fpath):
        for file_name in file_names:
            paths.append(join(root, file_name))
    return paths


def find_one(data_fpath='/home/aandronis/scikit_learn_data/lfw_home/lfw/'):
    folder_path = join(data_fpath, sorted(listdir(data_fpath))[0])
    return join(folder_path, listdir(folder_path)[0])


if __name__ == '__main__':
    # lfw_people = load_lfw_db('/home/aandronis/scikit_learn_data/lfw_home/lfw/')
    print(find_one())
