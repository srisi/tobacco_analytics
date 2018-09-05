
import multiprocessing as mp
import time

import numpy as np
import scipy
from tobacco.frequencies_preprocessing.preprocessing_collections import get_col_name_and_idx_dict
from tobacco.frequencies_preprocessing.preprocessing_doc_types import get_doc_types_to_idx_dict, \
    get_doc_types_doc_matrix
from tobacco.frequencies_preprocessing.preprocessing_filters import get_doc_type_filters
from tobacco.frequencies_preprocessing.preprocessing_totals import get_doc_type_totals_vectors
from tobacco.frequencies_preprocessing.preprocessing_years import multiply_and_transform_doc_to_year
from tobacco.utilities.sparse_matrices import csc_bool_to_np_cython, csc_to_np_int32

# 8/31/18: Why are these globals re-initialized? Here's I think the solution:
# add_doc_types_mp is accessed by a multiprocessing task. Hence, the globals can't be passed as variables.
DOC_TYPE_DOC_MATRIX = {
            'docs': get_doc_types_doc_matrix(docs_or_sections='docs'),                              # 100 MB    uint8
                'sections': get_doc_types_doc_matrix(docs_or_sections='sections')                   # 900 MB    uint8
            }
DOC_TYPES_AND_IDX_DICT = get_doc_types_to_idx_dict()

FILTERS = {
    'docs': get_doc_type_filters(return_type='csc', docs_or_sections='docs'),                       # 100 MB    bool
    'sections': get_doc_type_filters(return_type='csc', docs_or_sections='sections'),               # 1000 MB   bool
}

TOTALS = {
    'docs': get_doc_type_totals_vectors(docs_or_sections='docs', all_csc=True),                     #  180 MB   int32
    'sections': get_doc_type_totals_vectors(docs_or_sections='sections', all_csc=True)              # 1630 MB   int32
}

COLLECTIONS_AND_IDX_DICT = get_col_name_and_idx_dict()

DF_AGGREGATE = None
DF_ACTIVE_COLLECTION_FILTERS_NP = None


# multiprocessing globals
FLAG_ALL_DONE = "WORK FINISHED"
FLAG_WORKER_FINISHED_PROCESSING = "WORKER FINISHED PROCESSING"
NUMBER_OF_PROCESSES = 10


def add_doc_types_mp(df_aggregate, df_active_collection_filters_np, docs_or_sections, mp_results_queue):
    """ This subscript of calculate_ngrams_live uses multiprocessing to calculate the doc type data

    :param df_aggregate:
    :param df_active_collection_filters_np:
    :param docs_or_sections:
    :param mp_results_queue:
    :return:
    """

    global DF_AGGREGATE
    global DF_ACTIVE_COLLECTION_FILTERS_NP
    DF_AGGREGATE = df_aggregate
    DF_ACTIVE_COLLECTION_FILTERS_NP = df_active_collection_filters_np

    start = time.time()

    doc_types = []
    doc_type_groups = []

    results_queue = mp.Queue()

    # Doc type groups need to be processed first so add them before the individual document types to process.
    for doc_type_group in ['internal communication', 'marketing documents', 'internal scientific reports',
                           'news reports', 'scientific publications', 'court documents']:
        mp.Process(target=add_doc_types_mp_worker, args=(doc_type_group, docs_or_sections, results_queue)).start()

    # find the 9 most frequent document types to process
    doc_type_totals = DOC_TYPE_DOC_MATRIX[docs_or_sections] * df_aggregate
    no_doc_types = len(doc_type_totals)
    doc_type_totals = [(DOC_TYPES_AND_IDX_DICT[i], doc_type_totals[i]) for i in range(no_doc_types)]
    doc_type_totals.sort(key=lambda x:x[1], reverse=True)
    dt_groups = doc_type_totals[:9]

    for doc_type in dt_groups:
        doc_type_name, _ = doc_type
        mp.Process(target=add_doc_types_mp_worker, args=(doc_type_name, docs_or_sections, results_queue)).start()

    # process all 15 results (i.e. groups and individual document types)
    for i in range(15):
        new_result = results_queue.get()
        if not new_result:
            continue
        elif new_result['token'] in ['internal communication', 'marketing documents', 'internal scientific reports',
                                     'news reports', 'scientific publications', 'court documents']:
            doc_type_groups.append(new_result)
        else:
            doc_types.append(new_result)

    DF_AGGREGATE = None
    DF_ACTIVE_COLLECTION_FILTERS_NP = None

    print("doc types multi threaded took: {}".format(time.time() - start))
    mp_results_queue.put(('doc_types', doc_types))
    mp_results_queue.put(('doc_type_groups', doc_type_groups))


def add_doc_types_mp_worker(dt_name, docs_or_sections, results_queue):
    """ Process one document type and add the result to the local results_queue

    :param dt_name:
    :param docs_or_sections:
    :param results_queue:
    :return:
    """

    doc_type_filter = FILTERS[docs_or_sections][(dt_name, False)]
    if type(doc_type_filter) == scipy.sparse.csc.csc_matrix:
        doc_type_filter = csc_bool_to_np_cython(doc_type_filter)
    absolute = multiply_and_transform_doc_to_year(DF_AGGREGATE, doc_type_filter.view(dtype=np.uint8), docs_or_sections)
    total = int(np.sum(absolute))
    if total == 0:
        results_queue.put({})

    doc_type_totals = TOTALS[docs_or_sections][dt_name]
    if type(doc_type_totals) == scipy.sparse.csc.csc_matrix:
        doc_type_totals = csc_to_np_int32(doc_type_totals)
    doc_type_totals_years = multiply_and_transform_doc_to_year(doc_type_totals,
                                                       DF_ACTIVE_COLLECTION_FILTERS_NP, docs_or_sections) +1

    relative = absolute / doc_type_totals_years

    results_queue.put({
        'token': dt_name,
        'counts': absolute.tolist(),
        'frequencies': relative.tolist(),
        'total': total
    })



if __name__ == "__main__":
    pass