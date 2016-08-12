"""
L2 Sketch 1. For a point p in Rn, its L2 sketch is a bit
vector sigma(p) in {0, 1}m, with each bit sigma_i(p) produced by
sigma_i(p) = h_i(p) mod 2,
h_i(p) = A_i * p + b_i / W, for every i = 1,2, ..., m
A_i in Rn random vector with each dimension sampled independently from the standard Gaussian distribution N(0, 1)
b_i in R is sampled from the uniform distribution U[0,W).
W  window size
"""
try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import pickle
from os import path
from scipy.spatial.distance import euclidean, cosine


def generate_hash_vars(dim_size, window_size=500, bits=64):
    out_file = path.abspath(
        path.join(path.dirname(__file__), "data", "hash_vars_" + str(window_size) + "_" + str(bits) + ".p"))

    if path.isfile(out_file):
        print("Found hash vars. Loading...")
        return pickle.load(open(out_file, 'rb'))

    print("Generating hash vars...")

    hash_vars = []
    for i in range(bits):
        a_i = np.expand_dims(np.random.normal(0, 1, dim_size), axis=0)
        b_i = np.random.uniform(0, window_size, 1)[0]

        dim = {
            'a_i': a_i,
            'b_i': b_i,
        }

        hash_vars.append(dim)

    pickle.dump(hash_vars, open(out_file, 'wb'))
    return hash_vars


def generate_hash_maps(feature_maps=None, hash_vars=None, window_size=500, bits=64):
    hash_code_list = []
    counter = 0
    for feat_map in feature_maps:
        hash_list = []

        for i in range(bits):
            a_i = hash_vars[i]['a_i']
            b_i = hash_vars[i]['b_i']
            # print(a_i.shape)
            # print(p_i.shape)

            inner_product_ai_p = np.inner(feat_map, a_i)[0][0]
            # print(inner_product_ai_p)

            sigma = (inner_product_ai_p + b_i) / window_size
            # print(sigma)

            sigma_mod_2 = int(np.floor(np.mod(sigma, 2)))
            # print(sigma_mod_2)

            hash_list.append(str(sigma_mod_2))

        hash_code_list.append(''.join(hash_list))
        counter += 1
        if counter % 100 == 0:
            print("Generated " + str(counter) + " hash codes")

    return hash_code_list


def test_hashing():
    vector = pickle.load(open("data/feature_map.p", 'rb'))
    vector_2 = pickle.load(open("data/feature_map_2.p", 'rb'))
    vector_3 = pickle.load(open("data/feature_map_3.p", 'rb'))
    vector_4 = pickle.load(open("data/feature_map_4.p", 'rb'))
    vector_5 = pickle.load(open("data/feature_map_5.p", 'rb'))
    vector_6 = pickle.load(open("data/feature_map_6.p", 'rb'))
    vector_7 = pickle.load(open("data/feature_map_7.p", 'rb'))
    vector_8 = pickle.load(open("data/feature_map_8.p", 'rb'))

    feat_map_list = [vector, vector_2, vector_3, vector_4, vector_5, vector_6, vector_7, vector_8]

    img_hash_list = generate_hash_maps(feat_map_list)

    # for item in img_hash_list:
    #     print item


    def hamming2(s1, s2):
        """Calculate the Hamming distance between two bit strings"""
        assert len(s1) == len(s2)
        return sum(c1 != c2 for c1, c2 in zip(s1, s2))

    print("\nEuclidean Distances\n")
    print(euclidean(vector, vector_2), 'Antonis - antonis diagonal')
    print(euclidean(vector, vector_3), 'Antonis - antonis side')
    print(euclidean(vector, vector_4), 'Antonis - Nicola')
    print(euclidean(vector, vector_5), 'Antonis - elina face')
    print(euclidean(vector, vector_6), 'Antonis - elina side')
    print(euclidean(vector, vector_7), 'Antonis - nikos')
    print(euclidean(vector, vector_8), 'Antonis - napo')
    #
    # print("\nCosine Similarity Scores\n")
    # print(cosine(vector, vector_2), 'Antonis - antonis diagonal')
    # print(cosine(vector, vector_3), 'Antonis - antonis side')
    # print(cosine(vector, vector_4), 'Antonis - Nicola')
    # print(cosine(vector, vector_5), 'Antonis - elina face')
    # print(cosine(vector, vector_6), 'Antonis - elina side')
    # print(cosine(vector, vector_7), 'Antonis - nikos')
    # print(cosine(vector, vector_8), 'Antonis - napo')

    print("\nHamming distances\n")
    print(hamming2(img_hash_list[0], img_hash_list[1]), 'Antonis - antonis diagonal')
    print(hamming2(img_hash_list[0], img_hash_list[2]), 'Antonis - antonis side')
    print(hamming2(img_hash_list[0], img_hash_list[3]), 'Antonis - Nicola')
    print(hamming2(img_hash_list[0], img_hash_list[4]), 'Antonis - elina face')
    print(hamming2(img_hash_list[0], img_hash_list[5]), 'Antonis - elina side')
    print(hamming2(img_hash_list[0], img_hash_list[6]), 'Antonis - nikos')
    print(hamming2(img_hash_list[0], img_hash_list[7]), 'Antonis - napo ')


if __name__ == '__main__':
    test_hashing()
