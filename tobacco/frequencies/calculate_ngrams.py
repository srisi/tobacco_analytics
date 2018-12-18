from IPython import embed

import multiprocessing
import time

import numpy as np
from tobacco.frequencies_preprocessing.preprocessing_years_cython import transform_doc_to_year_array
from tobacco.frequencies_preprocessing.preprocessing_filters import get_active_filters_np

from tobacco.frequencies.calculate_ngrams_collections import add_collections_mp
from tobacco.frequencies.calculate_ngrams_doc_types import add_doc_types_mp
from tobacco.frequencies_preprocessing.preprocessing_globals_loader import get_globals
from tobacco.frequencies_preprocessing.preprocessing_search import parse_search_tokens
from tobacco.frequencies_preprocessing.preprocessing_tokens import get_tokens
from tobacco.frequencies_preprocessing.preprocessing_z_scores import get_z_scores


def get_frequencies(search_tokens, active_filters, globals, profiling_run=False):
    """ Processes one frequency query and returns the results as a dict

    :param search_tokens: unparsed search token string
    :param active_filters: dict of lists, e.g. {'doc_type': ["internal communication"], 'collection': [1,2],
                                                'availability': [], 'term': []}
    :param globals: the frequency globals
    :param profiling_run: Used to profile multiprocessing function. If true, they are run as a single process.
    :return: Dict consisting of 'data' and 'errors'


    Results include data for:
    - tokens (absolute counts, relative frequencies, z-scores)
    - collections (absolute counts and relative frequencies for 9 most frequent collections)
    - document type groups (absolute counts and relative frequencies)
    - document types (absolute counts and relative frequencies for 9 most frequent document types)

    Processing steps (* indicates run as separate process through multiprocessing)
    - *Parse search terms (run as separate process
    - Load overall, collections, and doc type filters
    - Load token vectors
    - *Calculate z-scores for tokens
    - *Add collections data
    - *Add document type data (including document type groups)

    >>> globals = get_globals()
    >>> active_filters = {'doc_type': [], 'collection': [], 'availability': [], 'term': []}
    >>> result = get_frequencies(['cancer', 'neuro*', 'carcin*'], active_filters, globals, profiling_run=False)

    """


    if len(active_filters['term']) == 0:
        docs_or_sections = 'docs'
    else:
        docs_or_sections = 'sections'

    print("Calculating Frequencies. Term filter is: {}. Using {}.".format(active_filters['term'], docs_or_sections))
    start_time = time.time()

    # Initialize multiprocessing queue to handle the results for the collections and document types
    mp_results_queue = multiprocessing.Queue()

    # parse the search tokens as a separate process...
    multiprocessing.Process(target=parse_search_tokens, args=(search_tokens, mp_results_queue)).start()

    # ... in the meantime, load the active doc type and collection filters.
    active_doc_type_filters_np, active_collection_filters_np, active_filters_np = get_active_filters_np(active_filters,
                                            globals['filters'], docs_or_sections=docs_or_sections, return_type=np.uint8)

    df = {'active_doc_type_filters_np': active_doc_type_filters_np,
          'active_collection_filters_np': active_collection_filters_np,
          'active_filters_np': active_filters_np}

    # create a total count per year array. Add 1 to totals to avoid division by 0 errors.
    df['totals_years'] = transform_doc_to_year_array(data=globals['totals']['totals'][docs_or_sections]['np'],
                                                filter=df['active_filters_np'], docs_or_sections=docs_or_sections) + 1

    # get the parsed search tokens. If there were errors, return them.
    token_list, token_search_errors = mp_results_queue.get()
    if len(token_list) == 0:
        print({'error': token_search_errors})
        return {'error': token_search_errors}

    # get the count data for all tokens.
    df = get_tokens(df, token_list, docs_or_sections)
    print("time tokens: {}".format(time.time() - start_time))


    # Second round of multiprocessing: calculate z-scores, collection and doc type data
    tokens_for_z_scores = [{'token': token['token'], 'counts': token['counts']} for token in df['tokens']]
    multiprocessing.Process(target=get_z_scores, args=(tokens_for_z_scores, df['totals_years'], mp_results_queue)).start()
    multiprocessing.Process(target=add_doc_types_mp, args=(df['aggregate'], df['active_collection_filters_np'],
                                                           docs_or_sections, mp_results_queue)).start()
    multiprocessing.Process(target=add_collections_mp, args=(df['aggregate'], df['active_doc_type_filters_np'],
                                                             docs_or_sections, mp_results_queue)).start()

    # for profiling purposes, make the multiprocessing parts use a single process
    # otherwise, profiling with the line profiler doesn't work.
    if profiling_run:
        test_queue = multiprocessing.Queue()
        add_collections_mp(df['aggregate'], df['active_doc_type_filters_np'], docs_or_sections, test_queue)
        cols = test_queue.get()

        add_doc_types_mp(df['aggregate'], df['active_collection_filters_np'], docs_or_sections, test_queue)
        doc_types_mp = test_queue.get()
        doc_type_groups_mp = test_queue.get()



    del df['aggregate']
    del df['aggregate_years']
    del df['totals_years']
    del df['active_filters_np']
    del df['active_collection_filters_np']
    del df['active_doc_type_filters_np']

    for i in range(4):
        print(i)
        mp_result = mp_results_queue.get()
        if mp_result[0] == 'z_scores':
            z_scores = mp_result[1]
            for token_id in range(len(z_scores)):
                df['tokens'][token_id]['z_scores'] = z_scores[token_id].tolist()
        else:
            df[mp_result[0]] = mp_result[1]


    for token_dict in df['tokens']:
        token_dict['counts'] = token_dict['counts'].tolist()
        token_dict['frequencies'] = token_dict['frequencies'].tolist()

    print("Time total: ", time.time() - start_time)

    embed()

    return {'data': df, 'error': token_search_errors}





if __name__ == "__main__":

    globals = get_globals()

    active_filters = {'doc_type': [], 'collection': [], 'availability': [],
                      'term': []}
    result = get_frequencies(['cancer', 'neuro*', 'carcin*'], active_filters, globals, profiling_run=False)

    try:
        print("success")
    except KeyError:
        print("key error")
        print(result)
