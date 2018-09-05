import gzip
import pickle

import numpy as np
from scipy.sparse import lil_matrix, csc_matrix
from tobacco.configuration import VALID_COLLECTIONS, PATH_TOKENIZED, DOC_COUNT, SECTION_COUNT
from tobacco.frequencies_preprocessing.preprocessing_sections import get_doc_id_to_section_id_dict
from tobacco.utilities.databases import Database
from tobacco.utilities.sparse_matrices import store_csr_matrix_to_file, load_csc_matrix_from_file


def get_col_name_and_idx_dict():
    """ Creates a dict of collection name to id as well as id to collection name

    For every id, code ('pm'), name:
    dict[id] = (code, name)
    dict[code] = id
    dict[name] = id

    :return:
    """

    try:
        col_name_and_idx_dict = pickle.load(gzip.open(PATH_TOKENIZED + 'col_name_and_idx_dict.pickle', 'rb'))

    except IOError:

        print("col_name_and_idx_dict not available. Creating now...")

        db = Database("TOB_RAW")
        con, cur = db.connect()

        cur.execute("SELECT id, code, name FROM idl_collection;")
        rows = cur.fetchall()

        col_name_and_idx_dict = {}

        for row in rows:
            col_name_and_idx_dict[row['id']] = {
                'code': row['code'],
                'name': row['name'],
                'name_short': row['name'].replace('Collection','').replace('Records', '')
                    .replace('(DATTA)', '').replace('&', 'and').strip()
            }
            col_name_and_idx_dict[row['code']] = row['id']
            col_name_and_idx_dict[row['name']] = row['id']

        print(col_name_and_idx_dict)

        pickle.dump(col_name_and_idx_dict, gzip.open(PATH_TOKENIZED + 'col_name_and_idx_dict.pickle', 'wb'))

    return col_name_and_idx_dict


def get_collection_doc_matrix(docs_or_sections='docs'):
    """
    Creates a transformation matrix (collection x doc) M such that
    M = transformation matrix (collection x doc)
    t = term vector
    x = vector of

    :param docs_or_sections: 'docs' or 'sections'
    :return:
    """

    try:
        collection_doc_matrix = load_csc_matrix_from_file(PATH_TOKENIZED + 'collection_doc_matrix_{}'.format(
            docs_or_sections))
        collection_doc_matrix = csc_matrix(collection_doc_matrix, dtype=np.uint8)

    except IOError:

        print("collection_doc_matrix not available for {}. Creating now...".format(docs_or_sections))

        db = Database("TOB_FULL")
        con, cur = db.connect()
        doc_id_to_section_id_dict = get_doc_id_to_section_id_dict()

        # 7/28/17: why the + 1 ?
        m = max([i for i in VALID_COLLECTIONS]) + 1

        n = DOC_COUNT
        if docs_or_sections == 'sections':
            n = SECTION_COUNT

        collection_doc_matrix = lil_matrix((m, n), dtype=np.float)

        print(collection_doc_matrix.shape)

        cur.execute("SELECT id, collection_id FROM docs;")

        while True:
            row = cur.fetchone()
            if not row:
                break

            doc_id = row['id']
            collection = row['collection_id']

            if doc_id % 10000 == 0:
                print(doc_id)

            if docs_or_sections == 'docs':
                collection_doc_matrix[collection, doc_id] = 1
            elif docs_or_sections == 'sections':
                first_id, final_id = doc_id_to_section_id_dict[doc_id]
                for section_id in range(first_id, final_id+1):
                    collection_doc_matrix[collection, section_id] = 1

        collection_doc_matrix = collection_doc_matrix.tocsc()
        store_csr_matrix_to_file(collection_doc_matrix, PATH_TOKENIZED + 'collection_doc_matrix_{}.npz'.format(
            docs_or_sections))


    return collection_doc_matrix


# def get_collection_preprocessing():
#
#     try:
#         data = pickle.load(gzip.open(PATH_TOKENIZED + 'collections.pickle', 'rb'))
#         col_name_and_idx_dict = data['col_name_and_idx_dict']
#         doc_collection_matrix = data['doc_collection_matrix']
#
#         print(doc_collection_matrix.shape)
#
#     except IOError:
#         print("Preprocessed collections not available. Generating now.")
#         col_name_and_idx_dict = get_col_name_and_idx_dict()
#         doc_collection_matrix = create_collection_doc_matrix()
#
#         data = {'col_name_and_idx_dict': col_name_and_idx_dict,
#                 'doc_collection_matrix': doc_collection_matrix}
#
#         pickle.dump(data, gzip.open(PATH_TOKENIZED + 'collections.pickle', 'wb'))
#
#     return col_name_and_idx_dict, doc_collection_matrix



if __name__ == "__main__":

    # get_collection_preprocessing()

    # get_collection_doc_matrix('docs')
    # get_collection_doc_matrix('sections')

    get_col_name_and_idx_dict()
#    create_col_name_to_idx_dict()