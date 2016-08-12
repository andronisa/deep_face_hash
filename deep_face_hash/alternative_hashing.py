import numpy as np


def _binary_array_to_hex(arr):
    """
    internal function to make a hex string out of a binary array
    """
    h = 0
    s = []
    for i, v in enumerate(arr.flatten()):
        if v:
            h += 2 ** (i % 8)
        if (i % 8) == 7:
            s.append(hex(h)[2:].rjust(2, '0'))
            h = 0
    return "".join(s)


class ImageHash(object):
    """
    Hash encapsulation. Can be used for dictionary keys and comparisons.
    """

    def __init__(self, binary_array):
        self.hash = binary_array

    def __str__(self):
        return _binary_array_to_hex(self.hash.flatten())

    def __repr__(self):
        return repr(self.hash)

    def __sub__(self, other):
        if other is None:
            raise TypeError('Other hash must not be None.')
        if self.hash.size != other.hash.size:
            raise TypeError('ImageHashes must be of the same shape.', self.hash.shape, other.hash.shape)
        return (self.hash.flatten() != other.hash.flatten()).sum()

    def __eq__(self, other):
        if other is None:
            return False
        return np.array_equal(self.hash.flatten(), other.hash.flatten())

    def __ne__(self, other):
        if other is None:
            return False
        return not np.array_equal(self.hash.flatten(), other.hash.flatten())

    def __hash__(self):
        # this returns a 8 bit integer, intentionally shortening the information
        return sum([2 ** (i % 8) for i, v in enumerate(self.hash.flatten()) if v])


def hex_to_hash(hexstr):
    """
    Convert a stored hash (hex, as retrieved from str(Imagehash))
    back to a Imagehash object.
    """
    l = []
    if len(hexstr) != 16:
        raise ValueError('The hex string has the wrong length')
    for i in range(8):
        h = hexstr[i * 2:i * 2 + 2]
        v = int("0x" + h, 16)
        l.append([v & 2 ** i > 0 for i in range(8)])
    return ImageHash(np.array(l))


def dhash(img_feat_map, hash_size=8):
    """ Compute a hash from a PIL *image*.
    Thanks to http://blog.iconfinder.com/detecting-duplicate-images-using-python/ . """

    feat_map = img_feat_map.transpose().flatten()
    new_ft_map = np.append(feat_map, np.zeros(34))
    pixels = new_ft_map.reshape((hash_size + 1, hash_size))
    # compute differences
    diff = pixels[1:, :] > pixels[:-1, :]
    hash_code = ImageHash(diff).__str__()

    return hash_code
