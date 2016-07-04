from __future__ import absolute_import
from batch import load_batch
from data_utils import get_file
import numpy as np
import os
import tarfile


def load_data():
	file_name = "orl_faces"
	file_path = os.path.abspath(os.path.dirname( __file__ ))

	path = get_file(file_name, origin=file_path, untar=True)

	nb_train_samples = 50000

	X_train = np.zeros((nb_train_samples, 3, 32, 32), dtype="uint8")
	y_train = np.zeros((nb_train_samples,), dtype="uint8")

	for i in range(1, 41):
		fpath = os.path.join(path, 's_' + str(i))
		data, labels = load_batch(fpath)
		print(data, labels)

		# X_train[(i-1)*10000:i*10000, :, :, :] = data
		# y_train[(i-1)*10000:i*10000] = labels
	exit()

	fpath = os.path.join(path, 'test_batch')
	X_test, y_test = load_batch(fpath)

	y_train = np.reshape(y_train, (len(y_train), 1))
	y_test = np.reshape(y_test, (len(y_test), 1))

	return (X_train, y_train), (X_test, y_test)

if __name__ == '__main__':
	load_data()