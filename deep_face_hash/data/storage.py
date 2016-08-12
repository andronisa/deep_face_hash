try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
from pymongo import MongoClient
from bson.binary import Binary
from pprint import pprint


def clear_collection():
    conn = MongoClient()
    collection = conn.test_database.random_arrays

    collection.remove()
    conn.close()


def store_to_mongo(faces):
    print("\n#################### MONGO INSERTION ##########################")
    conn = MongoClient()
    collection = conn.test_database.random_arrays

    bulk = collection.initialize_unordered_bulk_op()
    face_keys = ['Feature Map', 'Hash Code', 'Target', 'Name']

    for face in faces:
        face = list(face)
        bulk.insert(dict(zip(face_keys, face)))

    result = bulk.execute()
    pprint(result)
    conn.close()

    # bulk = db.test.initialize_unordered_bulk_op()
    # print("reading tolist()")
    # [pickle.loads(x['cpickle']) for x in collection.find()]

    # collection.insert({'cpickle': Binary(pickle.dumps(np.random.rand(50, 3), protocol=2))})
