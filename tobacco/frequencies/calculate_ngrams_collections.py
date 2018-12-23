
import multiprocessing as mp
import time
from IPython import embed
from tobacco.utilities.vector import Vector
import numpy as np
import scipy
from tobacco.configuration import VALID_COLLECTIONS
from tobacco.frequencies_preprocessing.preprocessing_collections import get_collection_doc_matrix, \
    get_col_name_and_idx_dict
from tobacco.frequencies_preprocessing.preprocessing_filters import get_collection_filters
from tobacco.frequencies_preprocessing.preprocessing_totals import get_collection_totals_vectors
#from tobacco.frequencies_preprocessing.preprocessing_years_cython import transform_doc_to_year_array
#from tobacco.utilities.vector_transformation import csc_bool_to_np_cython, csc_to_np_int32
from tobacco.utilities.multi_processor import MultiProcessor


# 8/31/18: I really don't understand why these globals aren't loaded through the globals loader. However, this is
# such an obvious thing to do that I will believe my 2017 self that there was a good reason for this choice.
# as an aside: I really don't know how these globals are available in all the subprocesses without eating up
# an absurd amount of memory...
# 8/31/18 Here's I think the solution:
# add_doc_types_mp is accessed by a multiprocessing task. Hence, the globals can't be passed as variables.

#COLLECTION_DOC_MATRIX = {
#                'docs': get_collection_doc_matrix(docs_or_sections='docs'),                 # 100 MB    uint8
#                'sections': get_collection_doc_matrix(docs_or_sections='sections')          # 770 MB    uint8
#            }

FILTERS = {
    'docs': get_collection_filters(return_type='mixed', docs_or_sections='docs'),           # 220 MB
#    'sections': get_collection_filters(return_type='csc', docs_or_sections='sections'),     # 750 MB
}
for filter in FILTERS['docs']:
    b = FILTERS['docs'][filter].sum

TOTALS = {
    'docs': get_collection_totals_vectors(docs_or_sections='docs'),                         # 81 MB     int32 csc
#    'sections': get_collection_totals_vectors(docs_or_sections='sections')                  # 690 MB    int32 csc
}

COLLECTIONS_AND_IDX_DICT = get_col_name_and_idx_dict()



@profile
def add_collections(df_aggregate, df_aggregate_csc, active_doc_type_filters_np, docs_or_sections, mp_results_queue):

#    df_aggregate_csc = df_aggregate.copy().convert_to_datatype('csc')

    # Sort filters by number of documents they represent
    filter_sums = []
    for filter_name in FILTERS[docs_or_sections]:
        if filter_name == ('msa_bat', False):
            continue
        filter = FILTERS[docs_or_sections][filter_name]
        if filter.sum > 0:
            filter_sums.append((filter_name, filter.sum))
    filter_sums_sorted = sorted(filter_sums, key=lambda x: x[1], reverse=True)

    # Select 9 collections with the most documents
    cols_filtered = []
    for filter_name, filter_sum in filter_sums_sorted:

        # if a filter's total is lower than the highest included filtered collection -> skip becaus
        # it has no chance of getting included.
        filter = FILTERS[docs_or_sections][filter_name]
        if len(cols_filtered) > 9 and cols_filtered[8]['total'] > filter_sum:
            continue

        embed()

        col_filtered = df_aggregate_csc.convert_to_year_array(filter_vec=filter)
        cols_filtered = cols_filtered[:9]

        cols_filtered.append({
            'name': filter_name[0],
            'absolute_counts': col_filtered,
            'total': col_filtered.sum
        })
        if len(cols_filtered) >= 9:
            cols_filtered = sorted(cols_filtered, key=lambda x: x['total'], reverse=True)


    cols_filtered = cols_filtered[:9]

    results = []

    for col in cols_filtered:
        name = col['name']
        collection_totals = TOTALS[docs_or_sections][name]
        collection_totals_filtered = collection_totals.convert_to_year_array(
            filter_vec=active_doc_type_filters_np)
        relative_frequencies = col['absolute_counts'] / collection_totals_filtered

        results.append({
            'token': COLLECTIONS_AND_IDX_DICT[name]['name_short'],
            'counts': col['absolute_counts'],
            'frequencies': relative_frequencies,
            'total': col['total']
        })
    for i in results: print(i)



def add_collections_mp(df_aggregate, df_active_doc_type_filters_np, docs_or_sections, mp_results_queue):
    """ This subscript of calculate_ngrams_live uses multiprocessing to calculate the collections data

    First, we identify the 9 collections with the most hits, then we start a subprocess to work on each one of them
    individually.

    :param df_aggregate: csc document or section vector with the summed counts from all search terms
    :param df_active_doc_type_filters_np: np uint8 array of the combined doc type filters
    :param docs_or_sections: "docs" or "sections"
    :param mp_results_queue: multiprocessing results queue
    :return:
    """

    global DF_AGGREGATE
    global DF_ACTIVE_DOC_TYPE_FILTERS_NP
    DF_AGGREGATE = df_aggregate
    DF_ACTIVE_DOC_TYPE_FILTERS_NP = df_active_doc_type_filters_np

    start = time.time()
    cols = []

    # find top 9 collections to look at individually.
    # calculate the totals for each collection
    cols_totals = COLLECTION_DOC_MATRIX[docs_or_sections] * df_aggregate
    cols_totals = [(i, cols_totals[i]) for i in VALID_COLLECTIONS]
    cols_totals.sort(key = lambda x: x[1], reverse=True)
    cols_groups = cols_totals[:9]

    results_queue = mp.Queue()

    for collection_id in cols_groups:
        collection_id = collection_id[0]
        p = mp.Process(target=add_collections_multi_worker, args=(collection_id, docs_or_sections, results_queue))
        p.start()

    for i in range(9):
        new_result = results_queue.get()
        if new_result == {} or new_result['total'] == 0:
            continue
        else:
            cols.append(new_result)

    DF_AGGREGATE = None
    DF_ACTIVE_DOC_TYPE_FILTERS_NP = None

    print("cols multi threaded took: {}".format(time.time() - start))
    mp_results_queue.put(('collections', cols))


def add_collections_worker(col_id, function_statics):

    embed()

def add_collections_multi_worker(col_id, docs_or_sections, results_queue):
    """ Process one collection and add the result to the local results_queue

    :param col_id:
    :param docs_or_sections:
    :param results_queue:
    :return:
    """

    collection_filter = FILTERS[docs_or_sections][(col_id, False)]
    if type(collection_filter) == scipy.sparse.csc.csc_matrix:
        collection_filter = csc_bool_to_np_cython(collection_filter)
    absolute_counts = transform_doc_to_year_array(data=DF_AGGREGATE, filter=collection_filter.view(dtype=np.uint8),
                                                  docs_or_sections=docs_or_sections)

    total = int(np.sum(absolute_counts))
    if total == 0:
        results_queue.put({})

    # calculate relative frequency of the search terms within the collection
    collection_totals = TOTALS[docs_or_sections][col_id]
    if type(collection_totals) == scipy.sparse.csc.csc_matrix:
        collection_totals = csc_to_np_int32(collection_totals)
    collection_totals_years = transform_doc_to_year_array(data=collection_totals, filter=DF_ACTIVE_DOC_TYPE_FILTERS_NP,
                                                          docs_or_sections=docs_or_sections) + 1
    relative_frequencies = absolute_counts / collection_totals_years

    results_queue.put({
        'token': COLLECTIONS_AND_IDX_DICT[col_id]['name_short'],
        'counts': absolute_counts.tolist(),
        'frequencies': relative_frequencies.tolist(),
        'total': total
    })

if __name__ == "__main__":



    df_aggregate = Vector().load_token_vector('addiction', return_type='np_int32')
    df_aggregate_csc = Vector().load_token_vector('addiction', return_type='csc')

    letter_filter = Vector().load_filter_vector('letter', 'doc_type')
    no_filter = Vector(np.ones(11303161, dtype=np.int32))
    queue = mp.Queue()
    add_collections(df_aggregate, df_aggregate_csc, no_filter, 'docs', queue)
#    df_active_doc_type_filters_np, docs_or_sections, mp_results_queue):
    pass
