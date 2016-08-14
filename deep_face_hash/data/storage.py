from pymongo import MongoClient
from pprint import pprint


def _get_collection_name(conn, hash_size=None):
    if hash_size:
        db = conn.deep_face_hash
        name = str(hash_size) + "_bit"

        return db[name], name
    return conn.test_database.random_arrays, "random_arrays"


def clear_collection(hash_size=None):
    conn = MongoClient()

    collection, col_name = _get_collection_name(conn, hash_size)

    print("Clearing Mongo Collection: " + col_name)
    collection.remove()
    conn.close()

    return True


def mongodb_store(faces, hash_size=None):
    print("\n#################### MONGO INSERTION ##########################")
    conn = MongoClient()

    collection, col_name = _get_collection_name(conn, hash_size)
    bulk = collection.initialize_unordered_bulk_op()

    face_keys = ['feature_map', 'hash_code', 'target', 'name']
    for face in faces:
        face = list(face)
        bulk.insert(dict(zip(face_keys, face)))

    print("Inserting to collection: " + col_name)
    result = bulk.execute()
    pprint(result)
    conn.close()

    return True


def mongodb_find(query, fields, lim=None, hash_size=None):
    conn = MongoClient()
    collection, col_name = _get_collection_name(conn, hash_size)
    print("Find from collection: " + col_name)

    result = list(collection.find(query, fields).limit(lim)) if lim else list(collection.find(query, fields))

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
