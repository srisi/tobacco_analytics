import numpy as np
from scipy.sparse import csc_matrix

from tobacco.configuration import PATH_TOKENIZED, VALID_COLLECTIONS
from tobacco.utilities.sparse_matrices import store_csr_matrix_to_file, load_csc_matrix_from_file
from tobacco.frequencies_preprocessing.preprocessing_filters import get_filters

from tobacco.frequencies_preprocessing.preprocessing_doc_types import get_dtype_dict
from tobacco.frequencies_preprocessing.preprocessing_totals_cython import create_totals_vector
from tobacco.frequencies_preprocessing.preprocessing_filters import get_active_filters_np
from tobacco.utilities.vector_transformation import csc_to_np_int32

from pathlib import Path
from tobacco.utilities.vector import Vector


def get_collection_totals_vectors(docs_or_sections='docs'):
    """ Get the csc totals vectors for every collection

    :param docs_or_sections: "docs" or "sections"
    :return:
    """

    totals = {}

    if docs_or_sections == 'docs':
        for i in VALID_COLLECTIONS:
            totals[i] = get_collection_totals_vector(i, docs_or_sections = 'docs', return_type='csc')
    elif docs_or_sections == 'sections':
        for i in VALID_COLLECTIONS:
            totals[i] = get_collection_totals_vector(i, docs_or_sections = 'sections', return_type='csc')

    else:
        raise ValueError('''valid values for docs_or_section in get_collection_totals_vectors are 'docs' and 'sections' 
                            but not {}.'''.format(docs_or_sections))

    return totals


def get_collection_totals_vector(collection_id, docs_or_sections, return_type='csc'):
    """ Load the totals vector for one collection

    :param collection_id: id of the collection
    :param docs_or_sections: "docs" or "sections"
    :param return_type: "csc" or "np" or csc sparse matrix or np array
    :return:
    """

    try:
        return Vector().load_totals_vector(collection_id, 'collection', docs_or_sections, 'csc')
#        csc = load_csc_matrix_from_file(PATH_TOKENIZED + 'totals/{}_{}'.format(collection_id, docs_or_sections))
    except IOError:

        print("Creating totals vector for collection, type: ", collection_id, docs_or_sections)

        filters = get_filters(return_type='np')
        totals_vector = csc_to_np_int32(get_totals_vector(docs_or_sections))
        _, active_collection_filters_np, _ = get_active_filters_np(
            active_filters={'doc_type': {}, 'collection': {collection_id}, 'availability': {}}, FILTERS=filters,
            docs_or_sections=docs_or_sections, return_type=np.uint8)

        filtered_dt = totals_vector * active_collection_filters_np
        # um... yeah... ugly but it creates the required mx1 sparse vector
        csc = csc_matrix(csc_matrix(filtered_dt).T, dtype=np.int64)
        store_csr_matrix_to_file(csc, PATH_TOKENIZED + 'totals/{}_{}'.format(collection_id, docs_or_sections))

    if return_type  == 'csc':
        return csc_matrix(csc, dtype=np.int32)
    else:
        return csc_to_np_int32(csc)


def get_doc_type_totals_vectors(docs_or_sections='docs', all_csc=True):

    totals = {}

    for dt in (get_dtype_dict()['valid'].union(set(get_dtype_dict()['groups'].keys()))):

        try:
#            file_name = 'totals/{}_{}'.format(dt.replace('/', '_'), docs_or_sections)
#            file_path = Path(PATH_TOKENIZED, file_name)
#            totals_vector = Vector().load_from_disk(file_path, return_type=return_type)
#            totals[dt] = totals_vector
            totals[dt] = load_csc_matrix_from_file(PATH_TOKENIZED + 'totals/{}_{}'.format(dt.replace('/', '_'), docs_or_sections))
        except IOError:

            # added 6/9/17. This is all very awkward (i.e. loading filters every time).
            # It replaces an older solution that used filter_vector(), which is no longer available.
            # Filters get loaded every time because otherwise, the filters would be held in memory twice.

            print("Creating doc type totals vector for: ", dt, docs_or_sections)
            filters = get_filters(return_type='np')
            totals_vector = csc_to_np_int32(get_totals_vector(docs_or_sections))
            active_doc_type_filters_np, _, _ = get_active_filters_np(
                active_filters={'doc_type': {dt}, 'collection': {}, 'availability': {}}, FILTERS=filters,
                docs_or_sections=docs_or_sections, return_type=np.uint8)

            filtered_dt = totals_vector * active_doc_type_filters_np
            # um... yeah... ugly but it creates the required mx1 sparse vector
            totals[dt] = csc_matrix(csc_matrix(filtered_dt).T, dtype=np.int64)

#            totals[dt] = filter_vector(get_totals_vector(), {'doc_type': {dt}, 'collection': {}})
            store_csr_matrix_to_file(totals[dt], PATH_TOKENIZED + 'totals/{}_{}'.format(dt.replace('/', '_'), docs_or_sections))


        if dt in {'court documents', 'scientific publications', 'internal scientific reports', 'news reports',
                  'marketing documents', 'internal communication'}:
            # somehow, parsing back to int64 is necessary. don't know why it's in int32 format

            if not all_csc:
                totals[dt] = csc_to_np_int32(csc_matrix(totals[dt], dtype=np.int64))
        if docs_or_sections == 'docs' and dt in {'report', 'letter', 'memo', 'email'}:
            if not all_csc:
                totals[dt] = csc_to_np_int32(csc_matrix(totals[dt], dtype=np.int64))

        if all_csc:
            totals[dt] = csc_matrix(totals[dt], dtype=np.int32)

    return totals



def get_totals_vector(docs_or_sections='docs', return_type='np_int32'):
    '''
    Only implemented for 1 gram because there's no reason why we would need totals for 2-5 grams
    :return:
    '''

    ngram = 1

    try:
        file_name = 'totals_{}_{}'.format(ngram, docs_or_sections)
        file_path = Path(PATH_TOKENIZED, file_name)
        totals_vector = Vector().load_from_disk(file_path, return_type=return_type)
        return totals_vector

    except IOError:

        totals_vector = create_totals_vector(ngram, docs_or_sections)
        totals_vector = csc_matrix(totals_vector, dtype=np.int32)
        store_csr_matrix_to_file(totals_vector, PATH_TOKENIZED + 'totals_{}_{}.npz'.format(ngram, docs_or_sections))
        return get_totals_vector(docs_or_sections, return_type)


if __name__ == "__main__":

#    get_doc_type_totals_vectors()

#    get_totals_vector()

#    get_collection_totals_vectors(return_type='np')
#    get_doc_type_totals_vectors()
#
#     a = {
#             'docs':     get_collection_totals_vectors(docs_or_sections='docs', return_type='np'),
#             'sections': get_collection_totals_vectors(docs_or_sections='sections', return_type='csc')
#         }
#     b = {
#             # 'docs':     get_doc_type_totals_vectors(docs_or_sections='docs'),
#             'sections': get_doc_type_totals_vectors(docs_or_sections='sections')
#         }

#    s = get_collection_totals_vectors('sections')
    test()
    pass
