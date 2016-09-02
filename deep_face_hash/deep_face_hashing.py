from __future__ import division
import dill
import pickle

import itertools
import time
import numpy as np
import matplotlib.pyplot as plt

from os import path
from data.img_utils import preprocess_images
from data.lfw_db import load_lfw_db, load_images
from data.storage import mongodb_store, mongodb_find, clear_collection
from bit_partition_lsh import generate_hash_maps, generate_hash_vars, calculate_mean_window_size
from vgg16 import generate_feature_maps, load_model
from utils import top_n_closer_vecs, top_n_closer_hash_codes, top_n_hamm_hash_codes, find_common_in_lists, nested_dict_from_two_list_combinations, nested_dict_from_three_list_combinations


def hash_lfw(fpath='/home/aandronis/scikit_learn_data/lfw_home/lfw/', reset_db=False,
			 existing_maps=False):
	start_time = time.time()
	bit_sizes = [64, 256, 1024, 4096]
	window_sizes = [2800, 3000]

	# Window 100-1000 --> bitsizes [64, 128, 256, 512, 1024]
	# Window 1000 - 2700 --- bitsizes [64, 128, 256, 512, 1024, 2048, 4096]

	# Current ---> 2700 for all
	# max_window_size+step is to include end (max_window_size)

	print("\n##################### INITIALIZE MODEL #########################")
	lfw_model = load_model()

	if not existing_maps:
		print("DELETED FEATURE MAPS COLLECTION! Re-preprocessing starting..")
		clear_collection('feature_maps')
	else:
		feature_maps = map(pickle.loads,
					   [item['feature_map'] for item in
						mongodb_find({}, {'feature_map': 1}, None, collection="feature_maps_final")])

	(chunked_img_paths, chunked_targets, chunked_names, img_options) = load_lfw_db(fpath)
	for window_size in window_sizes:
		for hash_size in bit_sizes:
			# compute one feature map to get dimensionality
			hash_vars = generate_hash_vars(model=lfw_model, window=window_size, bits=hash_size)

			# careful!!
			print("\n##################### DATABASE SETUP #########################")
			col_name = "_".join(("hash_maps", str(window_size), str(hash_size), "bit", "final"))

			if reset_db:
				clear_collection(col_name)

			print("\n#################### CHUNK HANDLING ##########################")
			print("\nStarting Hashing...")
			if existing_maps:
				hash_codes = generate_hash_maps(feature_maps, hash_vars, window_size, hash_size)
				targets = list(itertools.chain.from_iterable(chunked_targets))
				names = list(itertools.chain.from_iterable(chunked_names))
				batch_list = zip(hash_codes, targets, names)

				db_keys = ['hash_code', 'target', 'name']
				col_name = "_".join(("hash_maps", str(window_size), str(hash_size), "bit", "final"))
				mongodb_store(batch_list, db_keys, col_name)

				del targets
				del names
				del hash_codes
				del batch_list
			else:
				print("Total batches: " + str(img_options['n_batch']))
				print("Images per batch: " + str(img_options['num_faces']))

				batch_counter = 0
				for img_paths in chunked_img_paths:
					print("\nStarting image batch no." + str(batch_counter + 1) + "\n")
					print("Preprocessing Images...")

					preprocessed_images = preprocess_images(img_paths.tolist(), img_size=(224, 224), img_options=img_options)
					feature_maps = generate_feature_maps(preprocessed_images, lfw_model)
					hash_codes = generate_hash_maps(feature_maps, hash_vars, window_size, hash_size)
					batch_list = zip(hash_codes, chunked_targets[batch_counter].tolist(),
									 chunked_names[batch_counter].tolist())

					db_keys = ['hash_code', 'target', 'name']
					col_name = "_".join(("hash_maps", str(window_size), str(hash_size), "bit", "final"))
					mongodb_store(batch_list, db_keys, col_name)

					del preprocessed_images
					del feature_maps
					del hash_codes
					del batch_list

					print("\n##############################################")
					print("Finished image batch no." + str(batch_counter + 1))
					print("##############################################\n")

					batch_counter += 1

	print("--- %s seconds ---" % (time.time() - start_time))

	del chunked_img_paths
	del chunked_names
	del chunked_targets

	print("\n##############################################")
	print("Finished creation of hashcodes")
	print("##############################################\n")

	return True


def deep_face_hashing(fpath, print_names=False):
	start_time = time.time()
	bit_sizes = [64, 256, 1024, 4096]
	window_sizes = [2200, 2400, 2600, 2800, 3000]
	show_top_vecs = True

	print("\nStarting deep face hashing of a new image")
	print("\n##################### INITIALIZE MODEL #########################")
	lfw_model = load_model()

	img_paths = load_images(fpath)
	preprocessed_images = preprocess_images(img_paths, img_size=(224, 224))
	feature_maps = np.array(generate_feature_maps(preprocessed_images, lfw_model, insert=False))
	feat_map_collection = "feature_maps_final"
	lfw_feat_maps = np.array(map(pickle.loads,
								 [item['feature_map'] for item in
								  mongodb_find({}, {'feature_map': 1}, None, collection=feat_map_collection)]))
	lfw_feat_maps = lfw_feat_maps.reshape(lfw_feat_maps.shape[0], lfw_feat_maps.shape[2])

	for window_size in window_sizes:
		for hash_size in bit_sizes:
			# careful!!
			# print("\n##################### DATABASE SETUP #########################")
			col_name = "_".join(("hash_maps", str(window_size), str(hash_size), "bit", "final"))

			# compute one feature map to get dimensionality
			hash_vars = generate_hash_vars(model=lfw_model, window=window_size, bits=hash_size)
			hash_codes = generate_hash_maps(feature_maps, hash_vars, window_size, hash_size)
			names = np.array([item['name'] for item in mongodb_find({}, {'name': 1}, None, collection=col_name)])
			lfw_hash_maps = [item['hash_code'] for item in
							 mongodb_find({}, {'hash_code': 1}, None, collection=col_name)]

			if show_top_vecs:
				for feature_map in feature_maps:
					closest_indices = top_n_closer_vecs(lfw_feat_maps, feature_map, 10)
					print("\nTop 10 similar persons using feature map vectors:\n")

					top_ten_vecs = []
					for index in closest_indices:
						top_ten_vecs.append(names[index])
						print(names[index])
				show_top_vecs = False

			for hash_code in hash_codes:
				closest_indices = top_n_hamm_hash_codes(hash_code, lfw_hash_maps, 10)
				print("\nFor window size of: " + str(window_size) + " and hash size of: " + str(hash_size))
				print("Top 10 similar persons using hashmaps:\n")

				top_ten_hashes = []
				for index in closest_indices:
					top_ten_hashes.append(names[index])
					if print_names:
						print(names[index])

				common = find_common_in_lists(top_ten_vecs, top_ten_hashes)
				print("\nTotal common in top 10: " + str(len(common)))
				print(common)

	print("--- %s seconds ---" % (time.time() - start_time))
	return True


def generate_multiple_hash_map_collections():
	hash_lfw(reset_db=True, existing_maps=True)


def batch_test_lfw_hashes(print_names=True):
	start_time = time.time()
	bit_sizes = [64, 256, 1024, 4096]
	window_sizes = [2200, 2400, 2600, 2800, 3000]
	show_top_vecs = True

	(chunked_img_paths, chunked_targets, chunked_names, img_options) = load_lfw_db()
	paths = list(itertools.chain.from_iterable(chunked_img_paths))
	names = np.array(list(itertools.chain.from_iterable(chunked_names)))
	vals, indices, count = np.unique(names, return_counts = True, return_index=True)
	indices_to_avoid = indices[count == 1]
	names = names.tolist()

	del chunked_img_paths
	del chunked_targets
	del chunked_names

	feat_map_collection = "feature_maps_final"
	print("\nGetting the feature maps from the database...")
	lfw_feat_maps = map(pickle.loads,
								 [item['feature_map'] for item in
								  mongodb_find({}, {'feature_map': 1}, None, collection=feat_map_collection, pp=False)])

	feat_map_index = 0
	for feat_map in lfw_feat_maps:
		if feat_map_index in indices_to_avoid:
			feat_map_index += 1
			continue
		new_feat_maps = lfw_feat_maps
		# Calculation of closest feature maps
		new_feat_maps = np.array(new_feat_maps)
		new_feat_maps = new_feat_maps.reshape(new_feat_maps.shape[0], new_feat_maps.shape[2])
		vec_closest_indices = top_n_closer_vecs(new_feat_maps, feat_map, 11)[1:]

		printable_names = names
		printable_paths = paths

		current_name = printable_names[feat_map_index]
		current_path = printable_paths[feat_map_index]

		top_ten_vecs = []
		print("For photo: " + current_path)
		print("\nTop 10 similar persons to " + current_name + " using feature map vectors:\n")
		for index in vec_closest_indices:
			top_ten_vecs.append((printable_names[index], printable_paths[index]))
			print(printable_names[index], printable_paths[index])

		for window_size in window_sizes:
			for hash_size in bit_sizes:
				try:
					print("\nFor window size: " + str(window_size) + " - hash size: " + str(hash_size))
					# careful!!
					# print("\n##################### DATABASE SETUP #########################")
					col_name = "_".join(("hash_maps", str(window_size), str(hash_size), "bit", "final"))
					lfw_hash_maps = [item['hash_code'] for item in
									 mongodb_find({}, {'hash_code': 1}, None, collection=col_name, pp=False)]

					current_hash_code = lfw_hash_maps[feat_map_index]

					# Calculation of closest hash maps
					closest_indices = top_n_hamm_hash_codes(current_hash_code, lfw_hash_maps, 11)[1:]

					top_ten_hashes = []
					for index in closest_indices:
						top_ten_hashes.append((printable_names[index], printable_paths[index]))
					if print_names:
						for index in closest_indices:
							print(printable_names[index], printable_paths[index])

					common = find_common_in_lists(top_ten_vecs, top_ten_hashes)
					print("Total common in top 10: " + str(len(common)))
					print(common)

					del col_name
					del lfw_hash_maps
					del current_hash_code
					del closest_indices
					del top_ten_hashes
					del common
				except IndexError as er:
					print(er)
					continue
		del new_feat_maps
		del top_ten_vecs
		del vec_closest_indices
		del current_name
		del current_path
		del printable_names
		del printable_paths
		feat_map_index += 1
	print("--- %s seconds ---" % (time.time() - start_time))
	return True


def get_index(b):
	options = {
		64: 0,
		256: 1,
		1024: 2,
		4096: 3,
	}

	return options[b]


def top_result_accuracy():
	accurate = 0
	start_time = time.time()
	bit_sizes = [64, 256, 1024, 4096]
	window_sizes = [2200, 2400, 2600, 2800, 3000]
	show_top_vecs = True
	test_size = 100

	accuracies = nested_dict_from_two_list_combinations(window_sizes, bit_sizes)

	(chunked_img_paths, chunked_targets, chunked_names, img_options) = load_lfw_db()
	paths = list(itertools.chain.from_iterable(chunked_img_paths))
	names = np.array(list(itertools.chain.from_iterable(chunked_names)))
	vals, indices, count = np.unique(names, return_counts = True, return_index=True)
	indices_to_avoid = indices[count == 1]
	names = names.tolist()

	del chunked_img_paths
	del chunked_targets
	del chunked_names

	feat_map_collection = "feature_maps"
	print("\nGetting the feature maps from the database...")
	lfw_feat_maps = map(pickle.loads,
								 [item['feature_map'] for item in
								  mongodb_find({}, {'feature_map': 1}, None, collection=feat_map_collection, pp=False)])

	for window_size in window_sizes:
		for hash_size in bit_sizes:
			print("\nFor window size: " + str(window_size) + " - hash size: " + str(hash_size))
			accurate = 0
			counter = 0
			for feat_map_index in range(len(lfw_feat_maps)):
				if feat_map_index in indices_to_avoid:
					continue
				feat_map = lfw_feat_maps[feat_map_index]

				# Calculation of closest feature maps
				new_feat_maps = np.array(lfw_feat_maps)
				new_feat_maps = new_feat_maps.reshape(new_feat_maps.shape[0], new_feat_maps.shape[2])
				vec_closest_indices = top_n_closer_vecs(new_feat_maps, feat_map, 1)

				printable_names = names
				printable_paths = paths

				current_name = printable_names[feat_map_index]
				current_path = printable_paths[feat_map_index]

				top_ten_vecs = []
				for index in vec_closest_indices:
					top_ten_vecs.append((printable_names[index], printable_paths[index]))

				col_name = "_".join(("hash_maps", str(window_size), str(hash_size), "bit"))
				lfw_hash_maps = [item['hash_code'] for item in
								 mongodb_find({}, {'hash_code': 1}, None, collection=col_name, pp=False)]

				current_hash_code = lfw_hash_maps[feat_map_index]

				# Calculation of closest hash maps
				closest_indices = top_n_hamm_hash_codes(current_hash_code, lfw_hash_maps, 1)

				top_ten_hashes = []
				for index in closest_indices:
					top_ten_hashes.append((printable_names[index], printable_paths[index]))

				common = find_common_in_lists(top_ten_vecs, top_ten_hashes)
				common_size = len(common)
				accurate += common_size

				del col_name
				del lfw_hash_maps
				del current_hash_code
				del closest_indices
				del top_ten_hashes
				del common

				counter += 1

				if (counter) % 100 == 0:
					print("Calculated " + str(counter) + "accuracies")
				if (counter) % test_size == 0:
					break

			accuracies[window_size][get_index(hash_size)] = (accurate/test_size) * 100

			del(accurate)
			del new_feat_maps
			del top_ten_vecs
			del vec_closest_indices
			del current_name
			del current_path
			del printable_names
			del printable_paths

	print(accuracies)

	out_file = path.abspath(path.join(path.dirname(__file__), "data", "first_res_accuracy_" + str(test_size) + ".p"))
	pickle.dump(accuracies, open(out_file, "wb"))

	print("--- %s seconds ---" % (time.time() - start_time))
	return True


def top_ranked_accuracy(ranksize=10):
	bit_sizes = [64, 256, 1024, 4096]
	window_sizes = [2200, 2400, 2600, 2800, 3000]
	show_top_vecs = True
	test_size = 100
	# stop at 1000

	accuracies = nested_dict_from_three_list_combinations(window_sizes, bit_sizes, range(1, ranksize+1))

	(chunked_img_paths, chunked_targets, chunked_names, img_options) = load_lfw_db()
	paths = list(itertools.chain.from_iterable(chunked_img_paths))
	names = np.array(list(itertools.chain.from_iterable(chunked_names)))
	vals, indices, count = np.unique(names, return_counts = True, return_index=True)
	indices_to_avoid = indices[count == 1]
	names = names.tolist()

	del chunked_img_paths
	del chunked_targets
	del chunked_names
	del vals
	del indices
	del count

	feat_map_collection = "feature_maps"
	print("\nGetting the feature maps from the database...")
	lfw_feat_maps = map(pickle.loads,
								 [item['feature_map'] for item in
								  mongodb_find({}, {'feature_map': 1}, None, collection=feat_map_collection, pp=False)])

	for window_size in window_sizes:
		for hash_size in bit_sizes:
			col_name = "_".join(("hash_maps", str(window_size), str(hash_size), "bit"))
			lfw_hash_maps = [item['hash_code'] for item in
							 mongodb_find({}, {'hash_code': 1}, None, collection=col_name, pp=False)]

			print("\nFor window size: " + str(window_size) + " - hash size: " + str(hash_size) + "\n")
			counter = 0
			for feat_map_index in range(len(lfw_feat_maps)):
				try:
					if feat_map_index in indices_to_avoid:
						continue
					feat_map = lfw_feat_maps[feat_map_index]

					# Calculation of closest feature maps
					new_feat_maps = np.array(lfw_feat_maps)
					new_feat_maps = new_feat_maps.reshape(new_feat_maps.shape[0], new_feat_maps.shape[2])
					vec_closest_indices = top_n_closer_vecs(new_feat_maps, feat_map, ranksize+1)[1:]

					printable_names = names
					printable_paths = paths

					current_name = printable_names[feat_map_index]
					current_path = printable_paths[feat_map_index]

					top_ten_vecs = []
					for index in vec_closest_indices:
						top_ten_vecs.append((printable_names[index], printable_paths[index]))

					current_hash_code = lfw_hash_maps[feat_map_index]

					# Calculation of closest hash maps
					closest_indices = top_n_hamm_hash_codes(current_hash_code, lfw_hash_maps, ranksize+1)[1:]

					top_ten_hashes = []
					for index in closest_indices:
						top_ten_hashes.append((printable_names[index], printable_paths[index]))

					existing_ranks = {}
					exists = False
					for rank in range(1, ranksize+1):
						person_name, img = top_ten_hashes[rank-1]
						if (current_name == person_name):
							existing_ranks[rank] = person_name
							exists = True
					if exists:
						for existing, nam in existing_ranks.iteritems():
							accuracies[window_size][hash_size][existing-1][existing] += 1
						print("Found common for - " + current_name + " - on " + str(len(existing_ranks)) + " rank(s)")

					del current_hash_code
					del closest_indices
					del top_ten_hashes
					del existing_ranks
					del new_feat_maps
					del top_ten_vecs
					del vec_closest_indices
					del current_name
					del current_path
					del printable_names
					del printable_paths

					counter += 1

					if (counter) % 100 == 0:
						print("\nCalculated - " + str(counter) + " - accuracies\n")
					if (counter) % test_size == 0:
						break
				except IndexError as er:
					print(er)
					continue
			del col_name
			del lfw_hash_maps

			for rank in range(1, ranksize+1):
				accuracies[window_size][hash_size][rank-1][rank] = (accuracies[window_size][hash_size][rank-1][rank]/test_size)*100

	print(accuracies)

	out_file = path.abspath(path.join(path.dirname(__file__), "data", "old_accuracy_" + str(test_size) + ".p"))
	pickle.dump(accuracies, open(out_file, "wb"))

	for ws, hs_szs in accuracies.iteritems():
		for hs, scores in hs_szs.iteritems():
			rank_list = []
			acc_list = []
			for scoreset in scores:
				for rank, acc in scoreset.iteritems():
					rank_list.append(rank)
					acc_list.append(acc)
			plt.plot(rank_list, acc_list, 'ro')
			plt.plot(rank_list, acc_list, 'ro')
			plt.axis([1, 10, 0, 120])
			plt.show()
	del(accuracies)

	return True


def feature_rank_accuracy(ranksize=10):
	accurate = 0
	start_time = time.time()
	show_top_vecs = True
	test_size = 1000

	accuracies = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}

	(chunked_img_paths, chunked_targets, chunked_names, img_options) = load_lfw_db()
	paths = list(itertools.chain.from_iterable(chunked_img_paths))
	names = np.array(list(itertools.chain.from_iterable(chunked_names)))
	vals, indices, count = np.unique(names, return_counts = True, return_index=True)
	indices_to_avoid = indices[count == 1]
	names = names.tolist()

	del chunked_img_paths
	del chunked_targets
	del chunked_names
	del vals
	del indices
	del count

	feat_map_collection = "feature_maps_final"
	print("\nGetting the feature maps from the database...")
	lfw_feat_maps = map(pickle.loads,
								 [item['feature_map'] for item in
								  mongodb_find({}, {'feature_map': 1}, None, collection=feat_map_collection, pp=False)])

	counter = 0
	for feat_map_index in range(len(lfw_feat_maps)):
		if feat_map_index in indices_to_avoid:
			continue
		feat_map = lfw_feat_maps[feat_map_index]

		# Calculation of closest feature maps
		new_feat_maps = np.array(lfw_feat_maps)
		new_feat_maps = new_feat_maps.reshape(new_feat_maps.shape[0], new_feat_maps.shape[2])
		vec_closest_indices = top_n_closer_vecs(new_feat_maps, feat_map, ranksize+1)[1:]

		printable_names = names
		printable_paths = paths

		current_name = printable_names[feat_map_index]
		current_path = printable_paths[feat_map_index]

		top_ten_vecs = []
		for index in vec_closest_indices:
			top_ten_vecs.append((printable_names[index], printable_paths[index]))

		existing_ranks = {}
		exists = False
		for rank in range(1, ranksize+1):
			person_name, img = top_ten_vecs[rank-1]
			if (current_name == person_name):
				existing_ranks[rank] = person_name
				exists = True
		if exists:
			for existing, nam in existing_ranks.iteritems():
				accuracies[existing] += 1
			print("Found common for - " + current_name + " - on " + str(len(existing_ranks)) + " rank(s)")

		del top_ten_vecs
		del vec_closest_indices
		del new_feat_maps
		del current_name
		del current_path
		del printable_names
		del printable_paths

		counter += 1

		if (counter) % 100 == 0:
			print("\nCalculated " + str(counter) + " feat map accuracies\n")
		if (counter) % test_size == 0:
			break

	for rank in range(1, ranksize+1):
		accuracies[rank] = (accuracies[rank]/test_size)*100

	print(accuracies)

	out_file = path.abspath(path.join(path.dirname(__file__), "data", "feat_map_accuracy_" + str(test_size) + ".p"))
	pickle.dump(accuracies, open(out_file, "wb"))

	print("--- %s seconds ---" % (time.time() - start_time))
	return True


if __name__ == '__main__':
	# win_size = calculate_mean_window_size()
	# hash_lfw(fpath='/home/aandronis/scikit_learn_data/lfw_home/lfw/',
	#          reset_db=True,
	#          existing_maps=True)
	# deep_face_hashing(fpath='/home/aandronis/projects/deep_face_hash/deep_face_hash/data/img/test/', print_names=True)
	# batch_test_lfw_hashes(print_names=False)
	# top_result_accuracy()
	top_ranked_accuracy()
	# feature_rank_accuracy()
	# generate_multiple_hash_map_collections()

# [ 9666  6731  9569 12782  4295  1290 12044 13054 10929 12874]

# Top 10 similar persons to Tim Curry using feature map vectors:

# Paul Burrell
# Kate Winslet
# Alejandro Atchugarry
# Jacques Chirac
# Beecher Ray Kirby
# Denys Arcand
# Jorge Castaneda
# Giannina Facio
# Joseph Deiss
# Gerard Depardieu
