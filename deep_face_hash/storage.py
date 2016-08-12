try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
from pymongo import MongoClient
from bson.binary import Binary


def store_to_mongo():
    conn = MongoClient()
    collection = conn.test_database.random_arrays

    collection.remove()
    print("inserting with cpickle protocol 2")
    collection.insert({'cpickle': Binary(pickle.dumps(np.random.rand(50, 3), protocol=2))})
    # bulk = db.test.initialize_unordered_bulk_op()
    # print("reading tolist()")
    # [pickle.loads(x['cpickle']) for x in collection.find()]
