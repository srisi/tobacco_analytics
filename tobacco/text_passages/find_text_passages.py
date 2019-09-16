
import time
from collections import namedtuple

import numpy as np
from tobacco.frequencies_preprocessing.preprocessing_globals_loader import get_globals
from tobacco.text_passages.text_passages_helper_process_year_of_sections import process_year_of_sections_cython
from tobacco.text_passages.text_passages_helper_search import parse_text_passages_tokens
from tobacco.frequencies_preprocessing.preprocessing_filters import get_active_filters_np
from tobacco.utilities.ocr import load_vocabulary_trie

VOCABULARY = load_vocabulary_trie(1, return_type='set')

from tobacco.frequencies.calculate_ngrams_class import NgramResult


# only use end_year, not start_year

Document = namedtuple('Document', ['tid', 'title', 'date', 'year', 'collection'])
Passage = namedtuple('Passage', ['Document', 'text'])


def find_text_passages(tokens, active_filters, years_to_process, passage_length, globals,
                       logging=False, insert_result_to_db=True, max_no_docs_to_process=5000):
    """ This is the main task to find text passages matching one or more search terms.

    The main processing itself is done year by year in the cython function process_year_of_sections_cython

    :param tokens: string of tokens (as passed by website), e.g 'test, string'
    :param filters:
    :param start_year:
    :param end_year:
    :param passage_length:
    :param passages_per_year:
    :param min_readability:
    :param prepare_for_html: if True, titles are truncated, tids turned into urls, and search terms highlighted.
    :return:
    """

    print("getting here at least")



    filters = globals['filters']

    # output has errors (empty unless there is an error) and data list
    output_docs = {'errors': '', 'data': []}

    # split tokens and remove empty strings
    if len(tokens) == 0:
        output_docs['errors'] = "Please enter one or more search tokens in the text field."
        return output_docs

    first_token, search_regexes, token_intersection_vector = parse_text_passages_tokens(tokens)
    if logging:
        print(token_intersection_vector)
        print("first token, search regexes", first_token, search_regexes)

    if token_intersection_vector is None:
        # output_docs['errors'] =
        output_docs['errors'] = '{} does not exist in the vocabulary of tobacco-analytics.' \
                                ' Please enter a different term.'.format(tokens)
        return output_docs

    # 2019-01: Intermediate fix to get the filters
    _, _, active_filters_np = get_active_filters_np(active_filters, filters, docs_or_sections='sections')

    token_intersection_vector = token_intersection_vector.vector * active_filters_np.vector


    output_docs['sections'] = {}
    for year in years_to_process:
        output_docs['sections'][year] = process_year_of_sections_cython(first_token, tokens,
                            search_regexes, token_intersection_vector, year, passage_length,
                            active_filters, VOCABULARY, globals, insert_result_to_db,
                                                                    max_no_docs_to_process)

        if logging:
            try:
                print("\nSections for {}: {}".format(year, len(output_docs['sections'][year])))
                print(output_docs['sections'][year])
            except TypeError:
                print("\nSections for {}: 0".format(year))
                pass

    return output_docs



if __name__ == "__main__":


    globals = get_globals(globals_type='passages')
    active_filters = {'doc_type': ['internal communication'], 'collection': [6], 'availability': [], 'term': []}
    start = time.time()
    # results = find_text_passages('compound w', active_filters=active_filters, start_year=2000, end_year=2016, globals=globals, passage_length=200,
    #                     passages_per_year=20, min_readability=0.00, prepare_for_html=True)
    # print("Time", time.time() - start)

    start = time.time()
    results = find_text_passages(['youth smoking'], active_filters=active_filters,
                                 years_to_process=[i for i in range(1990, 1991)],
                                 globals=globals, passage_length=600, logging=True)

    print("no sections", sum([len(results['sections'][year]) for year in results]))
    print("Time", time.time() - start)

    from IPython import embed; embed()


    '''
    (998252, 998255)
    ('doc id: 128786. section_to_doc: [ 128786 5476484 5477605]', 1121)
    (-538, 'doc_section_start: 5475484. doc_section_end: 5474946')
    '''

