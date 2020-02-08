import multiprocessing
import json

import numpy as np
from IPython import embed

from tobacco.utilities.type_checker import check_param_type
from tobacco.frequencies_preprocessing.preprocessing_globals_loader import get_globals
from tobacco.frequencies_preprocessing.preprocessing_search import parse_search_tokens
from tobacco.frequencies_preprocessing.preprocessing_z_scores import get_z_scores
from tobacco.utilities.hash import generate_hash

from tobacco.configuration import SECTION_COUNT, DOC_COUNT
from tobacco.utilities.vector import Vector


# python "/home/stephan/anaconda3/envs/tobacco_analytics/lib/python3.6/site-packages/kernprof.py" -lv

class NgramResult():

    def __init__(self, doc_type_filters: list, collection_filters: list,
                 availability_filters: list, term_filters: list,
                 unparsed_search_tokens: list = None, parsed_search_tokens: list = None):

        self.unparsed_search_tokens = unparsed_search_tokens
        self.parsed_search_tokens = parsed_search_tokens
        self.errors = None
        self.docs_or_sections = None
        self.aggregate = None
        self.aggregate_years = None

        self.muliprocessing_queue = None


        # Results
        self.tokens_data = None
        self.collections = None
        self.doc_types = None
        self.doc_type_groups = None

        # Filters (These are inputs, lists of strings)
        self.doc_type_filters = doc_type_filters
        self.collection_filters = collection_filters
        self.availability_filters = availability_filters
        self.term_filters = term_filters
        self.active_filters = self._get_active_filters()


        # Filters (uint8 np arrays of the actual filters)
        self.doc_type_filters_np = None
        self.collection_filters_np = None
        self.availability_filters_np = None
        self.term_filters_np = None
        self.combined_filters_np = None


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


    def compute_result(self, globals):
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

        check_param_type(self.unparsed_search_tokens, list, 'unparsed_search_tokens',
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

        # ... in the meantime, load and set the active doc type, collection, availability,
        # term, and combined filters. They are stored as self.doc_type_filters_np,
        # self.combined_filters_np...
        self._compute_set_active_filters_np(globals)

        # create a total count per year array. Add 1 to totals to avoid division by 0 errors.
        totals_vector = globals['totals']['totals'][self.docs_or_sections]['np']
        self.totals_years = totals_vector.convert_to_year_array(filter_vec=self.combined_filters_np) + 1

        # get the parsed search tokens. If there were errors, return them.
        self.parsed_search_tokens, self.errors = mp_results_queue.get()
        if len(self.parsed_search_tokens) == 0:
            print({'error': self.errors})
            return {'error': self.errors}

        # get the count data for all tokens.
        # adds tokens_data (list of count/freq data for each token), aggregate (vector sum of all
        #  tokens), aggregate_years (aggregate as years vector)
        self._compute_add_tokens_data()

        # Second round of multiprocessing: calculate z-scores while adding collections and doc types
        multiprocessing.Process(target=get_z_scores, args=(self.tokens_data, self.totals_years,
                                                           mp_results_queue)).start()

        # add collections data
        self._compute_add_collection_data(globals)

        print("Collections")
        for i in self.collections:
            print(i)

        # add document type and document type group data
        self._compute_add_doc_type_data(globals)

        # release memory of variables that are no longer used
        self.aggregate = None
        self.aggregate_csc = None
        self.totals_years = None
        self.combined_filters_np = None
        self.collection_filters_np = None
        self.doc_type_filters_np = None


        mp_result = mp_results_queue.get()
        z_scores = mp_result[1]
        for token_id in range(len(z_scores)):
            self.tokens_data[token_id]['z_scores'] = z_scores[token_id].tolist()


    def _compute_set_active_filters_np(self, globals):

        """ Applies all filters to both the term and a copy of the totals vector and sets them

        All filters are np uint8 Vectors. The following filters are set in this function:
        doc_type_filters_np
        collection_filters_np
        availability_filters_np
        term_filters_np
        combined_filters_np

        6/10/17 Added availability filter.
        The idea is that every time the availability filter is used, the collection and doc type filters get multiplied with it.
        That is to say: the availability filter is not used on its own.

        7/25/17 Added term filter

        12/19/18: Moved to calculate_ngrams from preprocessing filters as this is a calculation
        step, not a pre-processing step

        >>> globals = get_globals(load_only_docs=True)
        >>> dt_filters = ['internal communication']
        >>> col_filters = [2,3]
        >>> avail_filters = ['no restrictions']
        >>> term_filters = []
        >>> search_tokens = ['addiction']
        >>> ngram = NgramResult(dt_filters, col_filters, avail_filters, term_filters, search_tokens)
        >>> ngram.docs_or_sections = 'docs'
        >>> ngram._compute_set_active_filters_np(globals=globals)
        >>> print(ngram.combined_filters_np)
        <Document Vector of type np_uint8 with 4964 elements.>

        :return: None
        """

        filters = globals['filters'][self.docs_or_sections]

        if not 'term' in self.active_filters:
            self.active_filters['term'] = {}

        # all filters used here are unweighted
        # 8/31/18: At some point, I had the idea that a document with 10 document types would give weight 1/10 to each.
        weighted = False
        filter_len = DOC_COUNT
        if self.docs_or_sections == 'sections':
            filter_len = SECTION_COUNT

        # process availability filters
        if len(self.active_filters['availability']) == 0:
            self.availability_filters_np = None
        else:
            self.availability_filters_np = None
            for filter_name in self.active_filters['availability']:
                if self.availability_filters_np is None:
                    self.availability_filters_np = filters['availability'][(filter_name, weighted)].copy()
                else:
                    self.availability_filters_np += filters['availability'][(filter_name, weighted)]
            self.availability_filters_np.convert_to_datatype('np_uint8')

        # process term filters
        if len(self.active_filters['term']) == 0:
            self.term_filters_np = None
        else:
            self.term_filters_np = None
            for filter_name in self.active_filters['term']:
                if self.term_filters_np is None:
                    self.term_filters_np = Vector().load_token_vector(filter_name, return_type='np_uint8',
                                                             docs_or_sections=self.docs_or_sections)
                else:
                    self.term_filters_np += Vector().load_token_vector(filter_name, return_type='np_uint8',
                                                              docs_or_sections=self.docs_or_sections)

        # process doc_type filters
        if len(self.active_filters['doc_type']) == 0:
            self.doc_type_filters_np = Vector(np.ones(filter_len, dtype=np.uint8))
        else:
            self.doc_type_filters_np = None
            for filter_name in self.active_filters['doc_type']:
                if self.doc_type_filters_np is None:
                    self.doc_type_filters_np = filters['doc_type'][(filter_name, weighted)].copy()
                else:
                    self.doc_type_filters_np += filters['doc_type'][(filter_name, weighted)]
            self.doc_type_filters_np.convert_to_datatype('np_uint8')

        # process collection filters
        if len(self.active_filters['collection']) == 0:
            self.collection_filters_np = Vector(np.ones(filter_len, dtype=np.uint8))
        else:
            self.collection_filters_np = None
            for filter_name in self.active_filters['collection']:
                if self.collection_filters_np is None:
                    self.collection_filters_np = filters['collection'][(filter_name, weighted)].copy()
                else:
                    self.collection_filters_np += filters['collection'][(filter_name, weighted)]
            self.collection_filters_np.convert_to_datatype('np_uint8')

        # Apply term filter to doc type and collection filters
        if self.term_filters_np is not None:
            self.doc_type_filters_np.filter_with(self.term_filters_np)
            self.collection_filters_np.filter_with(self.term_filters_np)

        # Apply availability filter to doc type and collection filters
        if self.availability_filters_np is not None:
            self.doc_type_filters_np.filter_with(self.availability_filters_np)
            self.doc_type_filters_np.filter_with(self.availability_filters_np)

        # Create final filter
        if len(self.active_filters['doc_type']) == 0:
            self.combined_filters_np = self.collection_filters_np
        elif len(self.active_filters['collection']) == 0:
            self.combined_filters_np = self.doc_type_filters_np
        else:
            self.combined_filters_np = self.collection_filters_np.filter_with(self.doc_type_filters_np,
                                                                   return_copy=True)

    def _compute_add_tokens_data(self):
        """
        Load counts, frequencies, and totals for each token.

        12/18/18 Moved from preprocessing_tokens and implemented for use with the NgramResult class,
        i.e. it won't return a df but individual vars.

        :return: None
        """

        self.tokens_data = []
        self.aggregate = None

        for token in self.parsed_search_tokens:
            # Load token and totals
            try:
                loaded_vector = Vector().load_token_vector(token, return_type='np_int32',
                                                           docs_or_sections=self.docs_or_sections)
            except FileNotFoundError:
                print('Could not load token {}.'.format(token))
                continue

            # initialize aggregate
            if self.aggregate is None:
                self.aggregate = loaded_vector.copy()
            else:
                self.aggregate += loaded_vector

            absolute_counts = loaded_vector.convert_to_year_array(filter_vec=self.combined_filters_np)

            self.tokens_data.append({
                'token': token,
                'counts': absolute_counts,
                'frequencies': absolute_counts / self.totals_years,
                'total': absolute_counts.sum
            })

        self.tokens_data = sorted(self.tokens_data, key=lambda k: k['total'], reverse=True)
        self.aggregate.filter_with(self.combined_filters_np)
        self.aggregate_csc = self.aggregate.copy().convert_to_datatype('csc')


    def _compute_add_collection_data(self, globals):

        # Sort filters by number of documents they represent
        filter_sums = []
        for filter_name in globals['filters'][self.docs_or_sections]['collection']:
            if filter_name == ('msa_bat', False):
                continue

            filter = globals['filters'][self.docs_or_sections]['collection'][filter_name]
            if filter.sum > 0:
                filter_sums.append((filter_name, filter.sum))
        filter_sums_sorted = sorted(filter_sums, key=lambda x: x[1], reverse=True)

        # Select 9 collections with the most documents
        cols_filtered = []
        for filter_name, filter_sum in filter_sums_sorted:

            # if a filter's total is lower than the highest included filtered collection -> skip becaus
            # it has no chance of getting included.
            filter = globals['filters'][self.docs_or_sections]['collection'][filter_name]
            if len(cols_filtered) > 9 and cols_filtered[8]['total'] > filter_sum:
                continue
            cols_filtered = cols_filtered[:9]

            col_filtered = self.aggregate_csc.convert_to_year_array(filter_vec=filter)


            cols_filtered.append({
                'name': filter_name[0],
                'absolute_counts': col_filtered,
                'total': col_filtered.sum
            })
#            embed()
            if len(cols_filtered) >= 9:
                cols_filtered = sorted(cols_filtered, key=lambda x: x['total'], reverse=True)

        cols_filtered = cols_filtered[:9]

        results = []

        for col in cols_filtered:
            name = col['name']
            collection_totals = globals['totals']['collection'][self.docs_or_sections][name]
            collection_totals_filtered = collection_totals.convert_to_year_array(
                filter_vec=self.doc_type_filters_np)
            relative_frequencies = col['absolute_counts'] / collection_totals_filtered

            results.append({
                'token': globals['collections_and_idx_dict'][name]['name_short'],
                'counts': col['absolute_counts'],
                'frequencies': relative_frequencies,
                'total': col['total']
            })

        self.collections = results


    def _compute_add_doc_type_data(self, globals):


        # Second, add all of the doc_type_groups
        dts = []
        for dt_group_name in ['internal communication', 'marketing documents',
                                    'internal scientific reports',
                                    'news reports', 'scientific publications', 'court documents']:
            dt = {'token': dt_group_name}
            dt_group_filter = globals['filters'][self.docs_or_sections]['doc_type'][(dt_group_name,
                                                                                     False)]
            agg_filtered_with_dt = self.aggregate_csc.convert_to_year_array(filter_vec=dt_group_filter)
            dt['absolute_counts'] = agg_filtered_with_dt
            dt['total'] = agg_filtered_with_dt.sum
            dt_group_totals = globals['totals']['doc_type'][self.docs_or_sections][dt_group_name]
            dt_group_totals_filtered = dt_group_totals.convert_to_year_array(
                filter_vec=self.collection_filters_np)
            freqs = dt['absolute_counts'] / dt_group_totals_filtered
            freqs.vector = np.nan_to_num(freqs.vector)
            dt['frequencies'] = freqs
            dts.append(dt)

        # Second, find the 9 most frequent document types to process
        dts_filtered = []
        for i in range(275):
            dt_name = globals['doc_type_and_idx_dict'][i]

            # 1/2019: the following dts are missing: 99 journal, 208 magazine,
            # 230 report - clinical study, 243 paper, 248 non printable/unable
            # 264 conference proceedings. Unclear why but these are small collections so it
            # shouldn't matter.
            try:
                dt_filter = globals['filters'][self.docs_or_sections]['doc_type'][(dt_name, False)]
            except:
                print(i, dt_name)
                continue
            dt_filter_sum = dt_filter.sum
            if len(dts_filtered) > 9 and dts_filtered[8]['total'] > dt_filter_sum:
                continue
            dts_filtered = dts_filtered[:9]

            agg_filtered_with_dt = self.aggregate_csc.convert_to_year_array(filter_vec=dt_filter)
            dts_filtered.append({
                'name': dt_name,
                'absolute_counts': agg_filtered_with_dt,
                'total': agg_filtered_with_dt.sum
            })
            if len(dts_filtered) >= 9:
                dts_filtered = sorted(dts_filtered, key=lambda x:x['total'], reverse=True)

        for dt in dts_filtered:
            dt_totals = globals['totals']['doc_type'][self.docs_or_sections][dt['name']]
            dt_totals_filtered = dt_totals.convert_to_year_array(filter_vec=
                                                                 self.collection_filters_np)
            freqs = dt['absolute_counts'] / dt_totals_filtered
            freqs.vector = np.nan_to_num(freqs.vector)
            dt['frequencies'] = freqs
            dts.append(dt)

        self.doc_types = dts



if __name__ == "__main__":
    unparsed_search_tokens = ['addic*']
    doc_type_filters = ['letter']
    collection_filters = [5,6,7]
    availability_filters = []
    term_filters = []

    doc_type_filters = []
    collection_filters = []

    globals = get_globals(load_only_docs=True)
    ngram = NgramResult(doc_type_filters, collection_filters, availability_filters,term_filters,
                        unparsed_search_tokens = unparsed_search_tokens)
    ngram.compute_result(globals)

