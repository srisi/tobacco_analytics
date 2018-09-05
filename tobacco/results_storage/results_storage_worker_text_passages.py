import gc
import json
import random
import time
import traceback
from collections import Counter

from tobacco.configuration import RESULTS_DB_NAME
from tobacco.frequencies_preprocessing.preprocessing_globals_loader import get_globals
from tobacco.results_storage.results_storage_redis import Redis_Con
from tobacco.text_passages.find_text_passages_mysql3 import find_text_passages
from tobacco.text_passages.text_passages_helper_distinctive_terms import get_text_passages_totals, \
    calculate_distinctive_terms
from tobacco.text_passages.text_passages_helper_topic_model import calculate_topic_model
from tobacco.utilities.databases import Database
from tobacco.utilities.email_notifications import send_email
from tobacco.utilities.hash import generate_hash

DB = Database(RESULTS_DB_NAME)
WORKERS_TO_RUN = 2

#VOCABULARY = load_vocabulary_trie(1)

GLOBALS_PASSAGES = get_globals(globals_type='passages')

REDIS_HOST = Redis_Con()


con_g, cur_g = DB.connect()

def look_for_text_passages_tasks_and_execute():
    """ This function is the interface between the redis task manager and the text passages search process

    If a text passage result needs to be calculated, this function activates the task to do so and adds the result
    to the results db.

    :return:
    """

    print("Text passages worker is ready")

    while True:

        task_type, task_params = REDIS_HOST.get_task_passages()
        print("task passages worker executes task {} with params: {}.".format(task_type, task_params))

        try:

            # we might only have to produce sections
            if task_type == 'complete':
                execute_full_text_passages_task(task_params)
            elif task_type == 'sections':
                tokens, active_filters, years_to_process, passage_length = task_params
                find_text_passages(tokens, active_filters, years_to_process, passage_length, GLOBALS_PASSAGES)
            else:
                raise ValueError("Only 'complete' and 'sections' are valid task_types for the passages worker but not: {}.".format(task_type))

            gc.collect()

        except:
            send_email('Text passages worker error',
                       '''Params: {}. Traceback: {}.'''.format(task_params, traceback.format_exc()))
            raise

def execute_full_text_passages_task(task_params):
    """ Executes a full text passages task, i.e. retrieve sections, distinctive terms, topic and store them in the db

    :param task_params:
    :return:
    """

    start = time.time()

    tokens, doc_type_filters, collection_filters, availability_filters, start_year, end_year, passage_length, \
    passages_per_year, min_readability = task_params
    active_filters = {'doc_type': doc_type_filters, 'collection': collection_filters, 'availability': availability_filters}
    query_hash = generate_hash((tokens, doc_type_filters, collection_filters, availability_filters, start_year, end_year,
                          passage_length, passages_per_year, min_readability))
    section_hash = generate_hash((tokens, doc_type_filters, collection_filters, availability_filters, passage_length))

    insert_cmd = '''REPLACE INTO results_passages (tokens, doc_type_filters, collection_filters, availability_filters,
                                                                start_year, end_year, passage_length, passages_per_year,
                                                                min_readability,
                                                                query_hash,
                                                                results, last_accessed, count_accessed)
                                        VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, DATE(NOW()), 0);'''

    con, cur = DB.connect()

    error_while_processing = False
    results_with_potential_error = None

    # Step 1: Find years to process
    years_to_process, years_to_load_from_db = get_years_to_extract_docs(tokens, doc_type_filters,
                                    collection_filters, availability_filters, passage_length, start_year, end_year, cur)

    # Step 2: If more than 20 years, split passage retrieval into multiple tasks and change years_to_load to whole range
    #         First chunk will be executed by this task
    if len(years_to_process) > 21:
        years_to_process_chunks = [years_to_process[i:i+21] for i in range(0, len(years_to_process), 21)]
        for years_to_process_chunk in years_to_process_chunks[1:]:
            REDIS_HOST.push_task_sections(('sections', tokens, active_filters, years_to_process_chunk, passage_length))
        years_to_load_from_db = list(range(start_year, end_year + 1))
        years_to_process = years_to_process_chunks[0]

    # Step 3 Process years if necessary and add to database
    print("step 3", years_to_process)
    if len(years_to_process) > 0:
        results_with_potential_error = find_text_passages(tokens, active_filters, years_to_process, passage_length,
                                                          GLOBALS_PASSAGES)
        if results_with_potential_error['errors'] != '':
            error_while_processing = True

    # Step 4: if errors while processing, store error
    if error_while_processing:
        cur.execute(insert_cmd, (str(tokens), str(doc_type_filters), str(collection_filters), str(availability_filters),
                                 start_year, end_year, passage_length, passages_per_year, min_readability, query_hash,
                                 json.dumps(results_with_potential_error)))
        con.commit()
        return

    # Step 5: Load raw sections if no errors
    if years_to_process == []:
        raw_sections = {}
    else:
        # if there are no sections, return empty dict
        try:
            raw_sections = results_with_potential_error['sections']
        except KeyError:
            raw_sections = {}
    if len(years_to_load_from_db) > 0:
        loaded_sections = load_sections(years_to_load_from_db, section_hash)
        raw_sections.update(loaded_sections)

    # Step 6 select and process sections
    final_sections, docs_by_year_list, section_id_to_year_dict = select_and_process_sections(raw_sections,
                                                             passages_per_year, min_readability, start_year, end_year)

    # Step 7 Distinctive authors, terms, topic models
    frequent_authors = get_frequent_authors(final_sections)

    text_passages_totals, top_2000_terms_set = get_text_passages_totals(final_sections, tokens)
    distinctive_terms, log_likelihoods = calculate_distinctive_terms(text_passages_totals, final_sections, tokens)


    #9/29: insert temporary result without topic model if more than 10000 docs
    if len(final_sections) > 10000:

        print("calculating topic model separately... ")

        temp_result = {'distinctive_terms': distinctive_terms,
                  'frequent_authors': frequent_authors,
                  'docs_by_year_list': docs_by_year_list,
                  'topic_model': {},
                  'errors' : ''}
        cur.execute(insert_cmd, (str(tokens), str(doc_type_filters), str(collection_filters), str(availability_filters),
                                 start_year, end_year, passage_length, passages_per_year, min_readability, query_hash,
                                 json.dumps(temp_result)))
        con.commit()

    # add all distinctive terms that weren't in the top 2000 terms
    for term in distinctive_terms:
        if term[0] not in top_2000_terms_set:
            top_2000_terms_set.add(term[0])



    topic_model_type = 'nmf'
    # in case I want to run nmf_sparse
    if passage_length == 601:
        topic_model_type = 'nmf_sparse'
    if passage_length == 599:
        topic_model_type = 'nmf_local_idf'
    topic_model = calculate_topic_model(final_sections, section_id_to_year_dict, vocabulary=list(top_2000_terms_set),
                                        log_likelihoods=log_likelihoods, n_components=20,
                                        topic_model_type=topic_model_type)

    # Step 8: Insert result
    result = {'distinctive_terms': distinctive_terms,
              'frequent_authors': frequent_authors,
              'docs_by_year_list': docs_by_year_list,
              'topic_model': topic_model,
              'errors' : ''}

    cur.execute(insert_cmd, (str(tokens), str(doc_type_filters), str(collection_filters), str(availability_filters),
                             start_year, end_year, passage_length, passages_per_year, min_readability, query_hash,
                             json.dumps(result)))
    con.commit()

    print("full passages task for {}, {}, {} took {}.\n".format(tokens, start_year, end_year, time.time() - start))



def load_sections(years_to_load_from_db, section_hash):

    # wait until sections are available

    years_unavailable = set(years_to_load_from_db)
    cur = None
    while True:
        _, cur = DB.connect()

        # this is inefficient. Better would be to just select the count and compare if all years are available...
        cur.execute('SELECT year FROM results_passages_yearly WHERE query_hash = "{}" AND year in ({});'.format(section_hash,
                                                                    ",".join([str(i) for i in years_to_load_from_db])))
        rows = cur.fetchall()
        if rows:
            for row in rows:
                if row['year'] in years_unavailable:
                    years_unavailable.remove(row['year'])
#                else:
#                    print("year unav error. year, unav, load", row['year'], years_unavailable, years_to_load_from_db)
        if len(years_unavailable) == 0:
            break
        else:
            # check if more sections to process before task can be finished
            _, section_task_params = REDIS_HOST.get_task_sections()
            if section_task_params:
                tokens, active_filters, years_to_process, passage_length = section_task_params
                print("Stalled while loading. Executing further sections task: {}".format(section_task_params))
                find_text_passages(tokens, active_filters, years_to_process, passage_length, GLOBALS_PASSAGES)
            else:
                time.sleep(0.1)

    loaded_sections = {}

    # then load all sections
    cur.execute('SELECT results, year FROM results_passages_yearly WHERE query_hash = "{}" AND year in ({});'.format(section_hash,
                                                                    ",".join([str(i) for i in years_to_load_from_db])))
    # load output sections and create docs_by_year_list
    while True:
        sections_row = cur.fetchone()
        if not sections_row: break
        year = sections_row['year']
        year_sections = json.loads(sections_row['results'].decode('utf-8'))
        loaded_sections[year] = year_sections

    return loaded_sections

def select_and_process_sections(raw_sections, passages_per_year, min_readability, start_year, end_year):

    docs_by_year_list = 116 * [0]
    output_sections = []
    section_id_to_year_dict = {}    # dict of final_section_id to year for topic model
    final_section_id = 0            # keep track of position in output_sections

    # initialize a set of of ids up to 5000 (max number of possible passages)
    # seed is set to passages_per_year so that the results remain constant but will change once passages_per_year changes.
    # shuffling is very cpu inefficient, so we're doing it only once.
    random.seed(passages_per_year)
    shuffled_section_ids = list(range(5000))
    random.shuffle(shuffled_section_ids)



    for year in range(start_year, end_year+1):
        print("year worker", year)

#        print("Year: {}, sections: {}.".format(year, len(raw_sections[year])))
        year_sections = raw_sections[year]
        if not year_sections or len(year_sections) == 0: continue
        for section_id in shuffled_section_ids:

           # if we have enough passages for the current year, break
            if docs_by_year_list[year-1901] == passages_per_year:
                break

            # in many cases, there are less than 5000 sections, so there will be many missed keys.
            try:
                section = year_sections[section_id]
            except IndexError:
                continue

            if section[6] > min_readability:
                output_sections.append(section)
                docs_by_year_list[year-1901] += 1
                section_id_to_year_dict[final_section_id] = year
                final_section_id += 1

    return output_sections, docs_by_year_list, section_id_to_year_dict


def get_years_to_extract_docs(tokens, doc_type_filters, collection_filters, availability_filters, passage_length,
                              start_year, end_year, cur):
    '''
    Create a list of years that were not previously processed
    '''

    years_to_process = list(range(start_year, end_year +1))
    section_hash = generate_hash((tokens, doc_type_filters, collection_filters, availability_filters, passage_length))

    cur.execute('SELECT year FROM results_passages_yearly WHERE query_hash = "{}";'.format(section_hash))
    for year_row in cur.fetchall():
        if year_row['year'] in years_to_process:
            years_to_process.remove(year_row['year'])

    # create a list of years to load from db
    years_to_load_from_db = list(range(start_year, end_year + 1))
    for year in years_to_process:
        years_to_load_from_db.remove(year)

    print("process: {}. \nload:{}.".format(years_to_process, years_to_load_from_db))

    return years_to_process, years_to_load_from_db


def get_frequent_authors(output_sections):

    authors = Counter()
    for section in output_sections:
        authors[section[3]] += 1

    authors[''] = 0

    # each author has name, count, yearly counts
    authors_selection = [[author[0], author[1], [0] * 116] for author in authors.most_common(10)]
    # maps author name to author_idx
    author_names = {author[0]:idx for idx, author in enumerate(authors_selection)}

    for section in output_sections:
        if section[3] in author_names:
            author_id = author_names[section[3]]
            year = section[2]
            authors_selection[author_id][2][year-1901] += 1




    return authors_selection

if __name__ == "__main__":

#    for i in range(WORKERS_TO_RUN):
#        multiprocessing.Process(target=look_for_text_passages_tasks_and_execute, args=()).start()
    look_for_text_passages_tasks_and_execute()

