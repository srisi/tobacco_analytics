from tobacco.utilities.vector import Vector
from tobacco.frequencies_preprocessing.preprocessing_globals_loader import get_globals
from tobacco.utilities.ocr import load_vocabulary_trie
from IPython import embed
import math
import sqlite3

from tobacco.utilities.databases import Database

GLOBAL_DOCS = get_globals(load_only_docs=True)
FILTERS = GLOBAL_DOCS['filters']['docs']
TOTALS_COL = GLOBAL_DOCS['totals']['collection']['docs']

COL_NAME_TO_ID = {
    'pm': 5,
    'rj': 6,
    'll': 7,
    'bw': 8,
    'at': 9,
    'ti': 10,
    'ba': 15,
#    'msa_bat': 'msa_bat'
}


def process_all():

    create_sqlite_table()

    terms = []

    db = Database('TOB_FULL')
    con, cur = db.connect()

    cur.execute('SELECT token from tokens where total > 10000;')
    for row in cur.fetchall():
        term = row['token']
        valid = True

        for word in term.split():
            if len(word) == 1:
                valid = False
            try:
                int(word)
                valid = False
            except ValueError:
                pass
        if valid:
            terms.append(term)

    print("Number of terms: {}".format(len(terms)))


    for collection in COL_NAME_TO_ID:
        col_id = COL_NAME_TO_ID[collection]
        filtered_collection_vector = FILTERS['doc_type'][('internal communication', False)].copy().convert_to_datatype('np_uint8')
        filtered_collection_vector.filter_with(FILTERS['collection'][(col_id, False)].convert_to_datatype('np_uint8'))
        max_5p_filter = Vector().load_page_number_vector('max_5')
        print("pre", filtered_collection_vector)
        filtered_collection_vector.filter_with(max_5p_filter)
        print('post', filtered_collection_vector)

        if collection == 'msa_bat':
            totals = TOTALS_COL[5]
            for id in [6, 7, 8, 9, 10, 11, 15]:
                totals += TOTALS_COL[id]
            print(totals)
        else:
            totals = TOTALS_COL[col_id]
        filtered_totals_year_vector = totals.convert_to_year_array(filter_vec=filtered_collection_vector)


        for term in terms:
            find_and_store_policy(term, filtered_collection_vector, filtered_totals_year_vector, collection)




def find_and_store_policy(term='and', filtered_collection_vector=None, filtered_totals_year_vector=None,
                          collection=None):


    db = sqlite3.connect('policies.db')
    cur = db.cursor()

    col_id = COL_NAME_TO_ID[collection]

    if not filtered_collection_vector:
        filtered_collection_vector = FILTERS['doc_type'][('internal communication', False)].copy().convert_to_datatype('np_uint8')
        filtered_collection_vector.filter_with(FILTERS['collection'][(col_id, False)].convert_to_datatype('np_uint8'))
        filtered_totals_year_vector = TOTALS_COL[col_id].convert_to_year_array(filter_vec=filtered_collection_vector)

    term_v = Vector().load_token_vector(token=term)
    term_year_vector = term_v.convert_to_year_array(filter_vec=filtered_collection_vector)


    dunnings = {}

    for start_first_period in range(50, 90):
        end_first_period = start_first_period + 3
        policy_year = start_first_period + 4
        start_second_period = start_first_period + 5
        end_second_period = start_first_period + 8

        term_count_first = term_year_vector[start_first_period : end_first_period+1].sum()
        term_count_second = term_year_vector[start_second_period : end_second_period+1].sum()

        totals_first = filtered_totals_year_vector[start_first_period : end_first_period+1].sum()
        totals_second = filtered_totals_year_vector[start_second_period : end_second_period+1].sum()

        dunning = dunning_log_likelihood(term_count_first, term_count_second,
                                         totals_first, totals_second)

        dunnings[policy_year] = {
            'year': f'19{policy_year}',
            'dunning': dunning,
            'first': term_count_first,
            'first_freq': term_count_first / totals_first*100,
            'second': term_count_second,
            'second_freq': term_count_second / totals_second*100
        }
#        print(f'19{start_first_period}-19{end_first_period} vs. 19{start_second_period}-'
#        f'19{end_second_period}: {dunning}. 1: {term_count_first}. 2: {term_count_second}')

    dunnings_sorted = sorted(dunnings.items(), key=lambda x:x[1]['dunning'])

    policy_adoption = dunnings_sorted[-1][1]
    policy_ending = dunnings_sorted[0][1]

    policy = '{}. {:15s}. Adoption: {}. D: {:7.0f}. C1: {:9d}. C2: {:9d}. F: {:5.3f}. ' \
             'Ending: {}. D:{:7.0f}. C1: {:9d}. C2: {:9d}. F: {:5.3f}.'.format( collection, term,
        policy_adoption['year'], policy_adoption['dunning'], policy_adoption['first'],
        policy_adoption['second'], policy_adoption['first_freq']/ policy_adoption['second_freq'],
        policy_ending['year'], policy_ending['dunning'], policy_ending['first'],
        policy_ending['second'], policy_ending['first_freq']/ policy_ending['second_freq']
    )
    print(policy)

    cur.execute('''INSERT INTO policies_5p VALUES("{}", "{}", 
                                               {}, {}, {}, {}, {}, {},
                                               {}, {}, {}, {}, {}, {})'''.format(
        collection, term,
        policy_adoption['year'], policy_adoption['dunning'], policy_adoption['first'],
        policy_adoption['second'], policy_adoption['first_freq'], policy_adoption['second_freq'],
        policy_ending['year'], policy_ending['dunning'], policy_ending['first'],
        policy_ending['second'], policy_ending['first_freq'], policy_ending['second_freq']
    ))
    db.commit()


def dunning_log_likelihood(term_c1:int, term_c2:int, totals_c1:int, totals_c2:int):

    a = float(term_c1) + 1
    b = float(term_c2) + 1
    c = totals_c1
    d = totals_c2

    e1 = c * (a + b) / (c + d)
    e2 = d * (a + b) / (c + d)
    dunning_log_likelihood = 2 * (a * math.log(a / e1) + b * math.log(b / e2))
    if a * math.log(a / e1) > 0:
        dunning_log_likelihood = -dunning_log_likelihood

    return dunning_log_likelihood


def find_distinctive_terms(term='younger adult smoker', collection='rj', start_year=1981,
                           end_year=1984):
    from tobacco.text_passages.find_text_passages import find_text_passages
    from tobacco.text_passages.text_passages_helper_distinctive_terms import get_text_passages_totals, \
        calculate_distinctive_terms
    from tobacco.results_storage.results_storage_worker_text_passages import select_and_process_sections

    col_id = COL_NAME_TO_ID[collection]
    globals = get_globals(globals_type='passages')
    active_filters = {'doc_type': ['internal communication'], 'collection': [col_id],
                      'availability': [], 'term': []}

    res = find_text_passages([term], active_filters=active_filters,
                             years_to_process=[i for i in range(start_year, end_year+1)],
                             globals=globals, passage_length=600, insert_result_to_db=False)

    raw_sections = res['sections']

    # Step 6 select and process sections
    final_sections, docs_by_year_list, section_id_to_year_dict = select_and_process_sections(raw_sections,
             passages_per_year=10000, min_readability=0.0, start_year=start_year, end_year=end_year)


    text_passages_totals, top_2000_terms_set = get_text_passages_totals(final_sections, [term])
    distinctive_terms, log_likelihoods = calculate_distinctive_terms(text_passages_totals,
                                                                     final_sections, [term])

    dt = {d[0]:d[2] for d in distinctive_terms}

    embed()



def create_sqlite_table():

    db = sqlite3.connect('policies.db')
    cur = db.cursor()

    cur.execute('''CREATE TABLE IF NOT EXISTS policies_5p (
                      collection    text,
                      term          text,
                      
                      pa_year       integer,
                      pa_dll        real,
                      pa_c1         integer,
                      pa_c2         integer,
                      pa_f1         real,
                      pa_f2         real,
                      
                      pe_year       integer,
                      pe_dll        real,
                      pe_c1         integer,
                      pe_c2         integer,
                      pe_f1         real,
                      pe_f2         real,
                      
                      UNIQUE (collection, term) ON CONFLICT REPLACE 
    );''')

    db.commit()





if __name__ == '__main__':
#    find_and_store_policy(term='younger adults', collection='rj')
    process_all()

#    find_distinctive_terms()


