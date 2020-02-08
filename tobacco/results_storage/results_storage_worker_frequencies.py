import json
import traceback

from MySQLdb import ProgrammingError
from tobacco.configuration import RESULTS_DB_NAME
from tobacco.frequencies.calculate_ngrams import get_frequencies
from tobacco.frequencies_preprocessing.preprocessing_globals_loader import get_globals
from tobacco.results_storage.results_storage_redis import Redis_Con
from tobacco.utilities.databases import Database
from tobacco.utilities.email_notifications import send_email
from tobacco.utilities.hash import generate_hash
from tobacco.frequencies.calculate_ngrams_class import NgramResult

GLOBALS_FREQUENCIES = get_globals(globals_type='frequencies')
DB = Database(RESULTS_DB_NAME)

REDIS_HOST = Redis_Con()


def look_for_frequencies_tasks_and_execute():
    """ This function is the interface between the redis task manager and the frequency calculation process

    If a frequency result needs to be calculated, this function activates the task to do so and adds the result
    to the results db.

    :return:
    """

    print("Frequencies worker is ready")


    while True:

        print("getting freq task")
        task = REDIS_HOST.get_task_frequencies()

        try:
            tokens, doc_type_filters, collection_filters, availability_filters, term_filters = task
            ngram_result = NgramResult(doc_type_filters=doc_type_filters,
                                       collection_filters=collection_filters,
                                       availability_filters=availability_filters,
                                       term_filters=term_filters,
                                       unparsed_search_tokens=tokens)
            ngram_result.compute_result(GLOBALS_FREQUENCIES)
            ngram_result.store_result_in_db(DB)

#            print("term filter in look_for_frequencies_tasks", term_filters, task)
#
#            active_filters = {'doc_type': doc_type_filters,
#                              'collection': collection_filters,
#                              'availability': availability_filters,
#                              'term': term_filters}
#            frequencies_result = get_frequencies(search_tokens=tokens,
#            # active_filters=active_filters,
#                                                 globals=GLOBALS_FREQUENCIES)
#            hash = generate_hash((tokens, doc_type_filters, collection_filters,
            # availability_filters, term_filters))
#            store_cmd = '''REPLACE INTO results_frequencies (tokens,
#                                                            doc_type_filters,
#                                                            collection_filters,
#                                                            availability_filters,
#                                                            term_filters,
#                                                            query_hash,
#                                                            results,
#                                                            last_accessed,
#                                                            count_accessed
#                                                            )
#                                        VALUES(%s, %s, %s, %s, %s, %s, %s, DATE(NOW()), 0);'''
#            con, cur = DB.connect()
#            cur.execute(store_cmd, (str(tokens), str(doc_type_filters), str(collection_filters),
#                                    str(availability_filters), str(term_filters), hash,
            # json.dumps(frequencies_result)))
#            con.commit()
#            con.close()

        except:


            # try:
            #     con.close()
            # except ProgrammingError:
            #     pass


            send_email('Frequencies worker error',
                       '''Task: {}. Stack Trace: {}'''.format(task, traceback.format_exc()))



            raise


if __name__ == '__main__':
    look_for_frequencies_tasks_and_execute()
#    look_for_tasks_and_execute()
