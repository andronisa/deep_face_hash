import numpy as np

from pymongo import MongoClient
from pprint import pprint


def _get_collection_name(conn, collection=None):
    if collection:
        db = conn.deep_face_hash
        return db[collection], collection
    return conn.test_database.random_arrays, "random_arrays"


def clear_collection(col_name=None):
    conn = MongoClient()

    collection, col_name = _get_collection_name(conn, col_name)

    print("Clearing Mongo Collection: " + col_name)
    collection.drop()
    conn.close()

    return True


def mongodb_store(items, keys=list(), collection=''):
    print("\n#################### MONGO INSERTION ##########################")
    conn = MongoClient()

    mongo_collection, col_name = _get_collection_name(conn, collection)
    bulk = mongo_collection.initialize_unordered_bulk_op()

    for item in items:
        if isinstance(item, tuple):
            to_insert = dict(zip(keys, list(item)))
        elif isinstance(item, list):
            to_insert = dict(zip(keys, item))
        else:
            to_insert = dict(zip(keys, [item]))

        bulk.insert(to_insert)
        del item
        del to_insert

    print("Inserting to collection: " + col_name)
    result = bulk.execute()
    pprint(result)

    conn.close()

    del items
    del bulk

    return True


def mongodb_find(query, fields, lim=None, collection=None, pp=True):
    conn = MongoClient()
    mongo_collection, col_name = _get_collection_name(conn, collection)
    if pp:
        print("Find from collection: " + col_name)

    result = list(mongo_collection.find(query, fields).limit(lim)) if lim else list(
        mongo_collection.find(query, fields))

    conn.close()

    return result


if __name__ == '__main__':
    # "_id", "name", "target", "hash_code" , "feature_map"

    # Debug find
    q = {}
    f = {'feature_map': 1}
    l = 1000
    res = mongodb_find(q, f, l)
    print(len(res))
