
import multiprocessing
import time
import json

import numpy as np
from IPython import embed

from tobacco.utilities.type_checker import check_param_type
from tobacco.frequencies_preprocessing.preprocessing_years import \
    transform_doc_to_year_array
from tobacco.frequencies_preprocessing.preprocessing_filters import get_active_filters_np

#from tobacco.frequencies.calculate_ngrams_collections import add_collections_mp
#from tobacco.frequencies.calculate_ngrams_doc_types import add_doc_types_mp
from tobacco.frequencies_preprocessing.preprocessing_globals_loader import get_globals
from tobacco.frequencies_preprocessing.preprocessing_search import parse_search_tokens
from tobacco.frequencies_preprocessing.preprocessing_tokens import get_tokens
from tobacco.frequencies_preprocessing.preprocessing_z_scores import get_z_scores
from tobacco.frequencies_preprocessing.preprocessing_tokens import get_ngram_vector
from tobacco.frequencies.calculate_ngrams_collections import add_collections_mp
from tobacco.frequencies.calculate_ngrams_doc_types import add_doc_types_mp
from tobacco.utilities.hash import generate_hash



class NgramResult():

    def __init__(self, doc_type_filters: list, collection_filters: list,
                 availability_filters: list, term_filters: list,
                 unparsed_search_tokens: list = None, parsed_search_tokens: list = None):

        self.unparsed_search_tokens = unparsed_search_tokens
        self.parsed_search_tokens = parsed_search_tokens
        self.aggregate = None
        self.aggregate_years = None

        self.muliprocessing_queue = None


        # Results
        self.tokens_data = None
        self.collections = None
        self.doc_types = None
        self.doc_type_groups = None

        # Filters
        self.doc_type_filters = doc_type_filters
        self.collection_filters = collection_filters
        self.availability_filters = availability_filters
        self.term_filters = term_filters

        self.active_collection_filters_np = None
        self.active_doc_type_filters_np = None

        self.active_filters = None
        self.active_filters_np = None


    def store_result_in_db(self, database):


        hash = generate_hash((self.parsed_search_tokens, self.doc_type_filters,
                              self.collection_filters, self.availability_filters, self.term_filters))
        store_cmd = '''REPLACE INTO results_frequencies (tokens,
                                                        doc_type_filters,
                                                        collection_filters,
                                                        availability_filters,
                                                        term_filters,
                                                        query_hash,
                                                        results,
                                                        last_accessed,
                                                        count_accessed
                                                        )
                                    VALUES(%s, %s, %s, %s, %s, %s, %s, DATE(NOW()), 0);'''
        con, cur = database.connect()
        cur.execute(store_cmd, (str(self.tokens), str(self.doc_type_filters),
                                str(self.collection_filters), str(self.availability_filters),
                                str(self.term_filters), hash,
                                json.dumps(self.generate_results_dict())))
        con.commit()
        con.close()

    def generate_results_dict(self):

        return {
            'error': self.errors,
            'data': {
                'tokens': self.tokens_data, # using 'tokens' for backwards compatibility
                'collections': self.collections,
                'doc_types': self.doc_types,
                'doc_type_groups': self.doc_type_groups
            }
        }





    def _get_active_filters(self):
        return {'doc_type': self.doc_type_filters, 'collection': self.collection_filters,
                'availability': self.availability_filters, 'term': self.term_filters}

    def compute_result(self, globals, profiling_run=False):
        """
        Computes the result for ngram

        >>> unparsed_search_tokens = ['addiction']
        >>> doc_type_filters = []
        >>> collection_filters = []
        >>> availability_filters = []
        >>> term_filters = []
        >>> globals = get_globals()
        >>> ngram = NgramResult(doc_type_filters, collection_filters, availability_filters,
        ...                     term_filters, unparsed_search_tokens=unparsed_search_tokens)
        >>> ngram.compute_result(globals)


        """

        check_param_type(self.unparsed_search_tokens, str, 'unparsed_search_tokens',
                         'NgramResult.__init__()')
        check_param_type(self.collection_filters, list, 'doc_type_filters',
                         'NgramResult.__init__()')
        check_param_type(self.availability_filters, list, 'availability_filters',
                         'NgramResult.__init__()')
        check_param_type(self.term_filters, list, 'term_filters', 'NgramResult.__init__()')
        check_param_type(self.doc_type_filters, list, 'doc_type_filters', 'NgramResult.__init__()')

        self.active_filters = self._get_active_filters()

        if len(self.term_filters) == 0:
            self.docs_or_sections = 'docs'
        else:
            self.docs_or_sections = 'sections'

        # Initialize multiprocessing queue to handle the results for the collections and document types
        mp_results_queue = multiprocessing.Queue()

        # parse the search tokens as a separate process...
        multiprocessing.Process(target=parse_search_tokens,
                                args=(self.unparsed_search_tokens, mp_results_queue)).start()

        # ... in the meantime, load the active doc type and collection filters.
        self.active_doc_type_filters_np, self.active_collection_filters_np, self.active_filters_np\
            = get_active_filters_np(
            self.active_filters, globals['filters'], docs_or_sections=self.docs_or_sections,
            return_type=np.uint8)

#        self.active_doc_type_filters_np = active_doc_type_filters_np
#        self.active_collection_filters_np = active_collection_filters_np
#        self.active_filters_np = active_filters_np

        # create a total count per year array. Add 1 to totals to avoid division by 0 errors.
        self.totals_years = transform_doc_to_year_array(
            data = globals['totals']['totals'][self.docs_or_sections]['np'],
            filter = self.active_filters_np,
            docs_or_sections = self.docs_or_sections
        ) + 1

        # get the parsed search tokens. If there were errors, return them.
        self.parsed_search_tokens, self.errors = mp_results_queue.get()
        if len(self.parsed_search_tokens) == 0:
            print({'error': self.errors})
            return {'error': self.errors}

        # get the count data for all tokens.
        # adds tokens_data (list of count/freq data for each token), aggregate (vector sum of all
        #  tokens), aggregate_years (aggregate as years vector)
        self._compute_add_tokens_data()

        # Second round of multiprocessing: calculate z-scores, collection and doc type data
        multiprocessing.Process(target=get_z_scores, args=(self.tokens_data, self.totals_years,
                                                           mp_results_queue)).start()
        multiprocessing.Process(target=add_doc_types_mp, args=(self.aggregate,
                                                               self.active_collection_filters_np,
                                                               self.docs_or_sections,
                                                               mp_results_queue)).start()
        multiprocessing.Process(target=add_collections_mp, args=(self.aggregate,
                                                                 self.active_doc_type_filters_np,
                                                                 self.docs_or_sections,
                                                                 mp_results_queue)).start()

        # for profiling purposes, make the multiprocessing parts use a single process
        # otherwise, profiling with the line profiler doesn't work.
        if profiling_run:
            test_queue = multiprocessing.Queue()
            add_collections_mp(self.aggregate, self.active_doc_type_filters_np,
                               self.docs_or_sections, test_queue)
            cols = test_queue.get()

            add_doc_types_mp(self.aggregate, self.active_collection_filters_np,
                             self.docs_or_sections, test_queue)
            doc_types_mp = test_queue.get()
            doc_type_groups_mp = test_queue.get()

        # release memory of variables that are no longer used
        self.aggregate = None
        self.totals_years = None
        self.active_filters_np = None
        self.active_collection_filters_np = None
        self.active_doc_type_filters_np = None

        for i in range(4):
            mp_result = mp_results_queue.get()
            if mp_result[0] == 'z_scores':
                z_scores = mp_result[1]
                for token_id in range(len(z_scores)):
                    self.tokens_data[token_id]['z_scores'] = z_scores[token_id].tolist()
            else:
                setattr(self, mp_result[0], mp_result[1])

        for token_dict in self.tokens_data:
            token_dict['counts'] = token_dict['counts'].tolist()
            token_dict['frequencies'] = token_dict['frequencies'].tolist()



    def _compute_add_tokens_data(self):
        """
        Load counts, frequencies, and totals for each token.

        12/18/18 Implemented for use with the NgramResult class, i.e. it won't return a df but
        individual vars.

        :param df: The final results dict
        :param token_list: list of tokens to process
        :param docs_or_sections: 'docs' or 'sections'
        :return: result df with added token data
        """

        self.tokens_data = []
        self.aggregate = None

        for token in self.token_list:
            # Load token and totals
            try:
                loaded_vector = get_ngram_vector(token, return_type='np',
                                                 docs_or_sections=self.docs_or_sections)

            except FileNotFoundError:
                print('Could not load token {}.'.format(token))
                continue

            # initialize aggregate
            if self.aggregate is None:
                self.aggregate = loaded_vector.copy()
            else:
                self.aggregate += loaded_vector

            absolute_counts = transform_doc_to_year_array(data=loaded_vector,
                                                          filter=self.active_filters_np,
                                                          docs_or_sections=self.docs_or_sections)

            self.tokens_data.append({
                'token': token,
                'counts': absolute_counts,
                'frequencies': absolute_counts / self.totals_years,
                'total': int(np.sum(absolute_counts))

            })

        self.tokens_data = sorted(self.tokens_data, key=lambda k: k['total'], reverse=True)

        self.aggregate *= self.active_filters_np
#        self.aggregate_years = transform_doc_to_year_array(data=self.aggregate,
#                                                      docs_or_sections=self.docs_or_sections)


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

    Sample Query:
    active_filters = {'doc_type': [], 'collection': [], 'availability': [], 'term': []}
    result = get_frequencies(['cancer', 'neuro*', 'carcin*'], active_filters, globals, profiling_run=False)
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
    return {'data': df, 'error': token_search_errors}





if __name__ == "__main__":
    unparsed_search_tokens = 'addiction'
    doc_type_filters = []
    collection_filters = []
    availability_filters = []
    term_filters = []
    globals = get_globals()
    ngram = NgramResult(doc_type_filters, collection_filters, availability_filters,term_filters, unparsed_search_tokens = unparsed_search_tokens)
    ngram.compute_result(globals)
    from IPython import embed; embed()
