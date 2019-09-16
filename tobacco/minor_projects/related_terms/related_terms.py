from tobacco.text_passages.find_text_passages import find_text_passages
from tobacco.frequencies_preprocessing.preprocessing_globals_loader import get_globals

import pickle
from collections import Counter


from IPython import embed

def find_related_terms(search_term):

    one_term_before_counter = Counter()
    two_terms_before_counter = Counter()
    one_term_after_counter = Counter()
    two_terms_after_counter = Counter()

    globals = get_globals(globals_type='passages')
    active_filters = {'doc_type': ['internal communication'], 'collection': [],
                          'availability': [], 'term': []}

    results = find_text_passages([search_term], active_filters=active_filters,
                                     years_to_process=[i for i in range(1940, 1998)],
                                     globals=globals, passage_length=200,
                                 insert_result_to_db=False, max_no_docs_to_process=10000000
                                 )

    for year in results['sections']:
        if results['sections'][year]:
            for section in results['sections'][year]:
                try:
                    passage = section[7].split()
                    term_idx = passage.index(f'<b>{search_term}</b>')

                    one_term_before = f'{passage[term_idx - 1]} {search_term}'
                    two_terms_before = f'{passage[term_idx - 2]} {one_term_before}'
                    one_term_after = f'{search_term} {passage[term_idx + 1]}'
                    two_terms_after = f'{one_term_after} {passage[term_idx + 1]}'

                    one_term_before_counter[one_term_before] += 1
                    two_terms_before_counter[two_terms_before] += 1
                    one_term_after_counter[one_term_after] += 1
                    two_terms_after_counter[two_terms_after] += 1
                except IndexError:
                    pass
                except ValueError:
                    pass

    print("done")
    embed()

if __name__ == '__main__':
    find_related_terms('youth')