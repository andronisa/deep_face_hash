import tarfile
import os
import sys


def get_file(fname, origin, untar=False):
	if not os.path.exists(origin):
		os.makedirs(origin)

	if untar:
		untar_fpath = os.path.join(origin, fname)
		fpath = untar_fpath + '.tar.gz'
	else:
		fpath = os.path.join(origin, fname)

	if untar:
		if not os.path.exists(untar_fpath):
			print('Untaring file...')
			tfile = tarfile.open(fpath, 'r:gz')
			try:
				tfile.extractall(path=origin)
			except (Exception, KeyboardInterrupt) as e:
				if os.path.exists(untar_fpath):
					if os.path.isfile(untar_fpath):
						os.remove(untar_fpath)
					else:
						shutil.rmtree(untar_fpath)
				raise
			tfile.close()
		return untar_fpath

	return fpath
