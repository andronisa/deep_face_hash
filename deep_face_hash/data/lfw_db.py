# http://scikit-learn.org/stable/auto_examples/applications/face_recognition.html
# Date - 05/07/2016

from __future__ import absolute_import
import numpy as np

from os.path import join, isdir, sys
from os import listdir, path

from img_utils import preprocess_images


# from sklearn.datasets import fetch_lfw_people


def load_lfw_db(data_fpath, batch_size=1000, img_size=None, crop_size=None, color_mode="rgb"):
    height = 250
    width = 250

    def chunks(l, n):
        """Yield successive n-sized chunks from l."""
        for i in range(0, len(l), n):
            yield l[i:i + n]

    chunked_image_path_list = list(chunks(sorted(listdir(data_fpath)), batch_size))

    for person_dir in chunked_image_path_list:
        person_names, file_paths = [], []
        for person_name in person_dir:
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

        img_options = {
            'height': height,
            'width': width,
            'num_faces': num_faces
        }

        preprocessed_images = preprocess_images(file_paths, img_size, crop_size, color_mode, img_options)

        # shuffle the faces with a deterministic RNG scheme to avoid having
        # all faces of the same person in a row, as it would break some
        # cross validation and learning algorithms such as SGD and online
        # k-means that make an IID assumption

        indices = np.arange(num_faces)
        np.random.RandomState(42).shuffle(indices)
        faces, target, person_names = preprocessed_images[indices], target[indices], person_names[indices]

        return faces, target, person_names
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