try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np

from collections import Counter as mset
from bson.binary import Binary
from sklearn.metrics.pairwise import pairwise_distances


def find_common_in_lists(list_1, list_2):
    intersection = mset(list_1) & mset(list_2)
    return list(intersection.elements())


def arr_to_binary(np_arr):
    return Binary(pickle.dumps(np_arr, protocol=2))


def hamming_distance(s1, s2):
    """Calculate the Hamming distance between two bit strings"""
    assert len(s1) == len(s2)
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def top_n_hamm_hash_codes(s1, list_s, top_n):
    distances = np.array([hamming_distance(s1, s) for s in list_s])
    distances = np.append(distances, [])

    # Sort the distances by value(smaller first) and keep the 10 first
    ten_smallest_indices = distances.argsort()[:top_n]
    print("")
    for index in ten_smallest_indices:
        print(distances[index])

    return ten_smallest_indices


def top_n_closer_hash_codes(s1, list_s, top_n):
    s1_vec = np.array([int(i) for i in s1]).reshape(1, -1)
    s_arr = np.array([[int(i) for i in s2] for s2 in list_s])

    return top_n_closer_vecs(s1_vec, s_arr, chosen_metric='hamming')

    # exit()
    # """Calculate the Hamming distance between two bit strings"""
    # assert len(s1) == len(s2)
    # return sum(c1 != c2 for c1, c2 in zip(s1, s2))


def top_n_closer_vecs(v1, v2, top_n=10, chosen_metric='euclidean'):
    # # Debug
    # print(v1.shape)
    # print(v2.shape)

    distances = pairwise_distances(v1, v2, metric=chosen_metric)
    distances = np.append(distances, [])

    # print(distances)

    # Sort the distances by value(smaller first) and keep the 10 first
    ten_smallest_indices = distances.argsort()[:top_n]
    print("")
    for index in ten_smallest_indices:
        print(distances[index])

    return ten_smallest_indices

# def top_n_closer_hashes(hash_code, hash_maps, top_n=10):
#     # # Debug
#     # print(v1.shape)
#     # print(v2.shape)
#
#     print(type(hash_code))
#     print(type(hash_maps))
#     hamming_distance_vecs(hash_code, hash_maps)
#     exit()
#
#     distances = pairwise_distances(v1, v2)
#     distances = np.append(distances, [])
#
#     print(distances)
#
#     # Sort the distances by value(smaller first) and keep the 10 first
#     ten_smallest_indices = distances.argsort()[:top_n]
#
#     for index in ten_smallest_indices:
#         print(distances[index])
#
#     return ten_smallest_indices
