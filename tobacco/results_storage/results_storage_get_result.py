import json
import os
import random
import re
import time
import traceback
from collections import Counter

from MySQLdb import ProgrammingError

from tobacco.configuration import RESULTS_DB_NAME
from tobacco.results_storage.results_storage_redis import Redis_Con
from tobacco.utilities.databases import Database
from tobacco.utilities.email_notifications import send_email
from tobacco.utilities.hash import generate_hash

DB = Database(RESULTS_DB_NAME)
REDIS_CON = Redis_Con()

SANITIZE_REGEX = re.compile(r'[^a-zA-Z0-9*,\ \-="\']+')



def get_frequencies_results(tokens, doc_type_filters, collection_filters, availability_filters, term_filters):
    """ Loads the results for one frequency query.

    First checks if they are already in the database. If not, it adds the task to the redis queue and waits for the
    result to be added to the database

    :param tokens:
    :param doc_type_filters:
    :param collection_filters:
    :param availability_filters:
    :param term_filters:
    :return:
    """

    try:

        tokens, doc_type_filters, collection_filters, availability_filters, term_filters = sanitize_query_input(
            tokens, doc_type_filters, collection_filters, availability_filters, term_filters
        )

        result = None

        con, cur = DB.connect()

        hash = generate_hash((tokens, doc_type_filters, collection_filters, availability_filters, term_filters))

        cur.execute('SELECT results, count_accessed FROM results_frequencies WHERE query_hash = "{}";'.format(hash))
        rows = cur.fetchall()

        # if results was precalculated
        if len(rows) > 0:
            result = rows[0]['results']

        # else, calculate and store the result
        else:
            REDIS_CON.push_task_frequencies((tokens, doc_type_filters, collection_filters, availability_filters, term_filters))

            count = 0
            while True:
                if count == 3000:
                    print("did not find result after 20000 loops or 10 minutes for {}".format(tokens, doc_type_filters, collection_filters, availability_filters, term_filters))
                    result = {'errors': "Something went wrong with this query and we terminated the calculation after 5 minutes."}
                    send_email('Frequencies failure', '''Tokens: {}. Doc Type Filters: {}. Collection Filters: {}.
                                Availability Filters: {}. Term Filters: {}.'''.format(tokens, doc_type_filters, collection_filters,
                                                                                      availability_filters, term_filters))
                    con.close()
                    return result
                con, cur = DB.connect()
                cur.execute('SELECT results, count_accessed FROM results_frequencies WHERE query_hash = "{}";'.format(hash))
                print(os.getpid(), count, 'frequencies', 'SELECT results, count_accessed FROM results_frequencies WHERE query_hash = "{}";'.format(hash))
                rows = cur.fetchall()
                if len(rows) == 0:
                    count += 1
                    time.sleep(0.1)
                    continue
                else:
                    result = rows[0]['results']
                    break

        result = json.loads(result.decode('utf-8'))


        count_accessed = rows[0]['count_accessed'] + 1
        cur.execute('UPDATE results_frequencies SET count_accessed = {} WHERE query_hash = "{}";'.format(count_accessed, hex))
        con.commit()
        con.close()

        return result

    except:

        send_email('Frequencies error',
                   '''Tokens: {}. Doc Type Filters: {}. Collection Filters: {}.
                    Availability Filters: {}. Term Filters: {}.
                    Stack Trace: {}'''.format(tokens, doc_type_filters, collection_filters, availability_filters, term_filters,
                                              traceback.format_exc()))

        try:
            con.close()
        except ProgrammingError:
            pass

        raise


def get_text_passages_results(tokens, doc_type_filters, collection_filters, availability_filters,
                              start_year, end_year, passage_length, passages_per_year, min_readability):
    """ This is the main task set up to retrieve text passages results.

    It checks if the result is already available in the database. If not, it adds the task to the redis queue
    Sections are loaded separately through the get_sections function.

    :param tokens:
    :param doc_type_filters:
    :param collection_filters:
    :param availability_filters:
    :param start_year:
    :param end_year:
    :param passage_length:
    :param passages_per_year:
    :param min_readability:
    :return:
    """

    try:

        start_time = time.time()
        print("Getting text passages result", tokens, start_year, end_year)

        tokens, token_parse_errors, doc_type_filters, collection_filters, availability_filters = sanitize_query_input(
            tokens, doc_type_filters, collection_filters, availability_filters, type='passages'
        )

        con, cur = DB.connect()
        query_hash = generate_hash((tokens, doc_type_filters, collection_filters, availability_filters, start_year,
                                    end_year, passage_length, passages_per_year, min_readability))


        cur.execute('SELECT results, count_accessed FROM results_passages WHERE query_hash = "{}"'.format(query_hash))
        rows = cur.fetchall()

        # if result available, just load
        if len(rows) > 0:
            results = json.loads(rows[0]['results'].decode('utf-8'))

            try:
                print("errors", results['errors'], 'errors' in results)
            except:
                pass


            # only load sections if there are no errors
            if 'errors' in results and len(results['errors']) > 0:
                results['sections'] = []
            else:
                results['sections'] = get_sections(tokens, doc_type_filters, collection_filters, availability_filters,
                       passage_length, passages_per_year, start_year, end_year, min_readability)

            count_accessed = rows[0]['count_accessed'] + 1

        # else, create new task, load section, then load overall results
        else:
            print("result not available, creating task")
            REDIS_CON.push_task_passages(('complete', tokens, doc_type_filters, collection_filters, availability_filters,
                                            start_year, end_year, passage_length, passages_per_year, min_readability))
            results = dict()

            count = 0
            while True:
                if count == 3000:
                        print("did not find result after 3000 loops or 5 minutes for {}".format(tokens, doc_type_filters,
                                                                            collection_filters, availability_filters))
                        result = {'errors': "Something went wrong with this query and we terminated the calculation after 5 minutes."}
                        send_email('Text passages failure', '''Tokens: {}. Doc Type Filters: {}. Collection Filters: {}.
                                    Availability Filters: {}.
                                    Start Year: {}. End Year: {}. Passages Length: {}. Passages per Year: {}. Min Readability: {}.'''.format(
                                    tokens, doc_type_filters, collection_filters, availability_filters,
                                    start_year, end_year, passage_length, passages_per_year, min_readability))
                        con.close()
                        return result

                con, cur = DB.connect()
                cur.execute('SELECT results, count_accessed FROM results_passages WHERE query_hash = "{}";'.format(query_hash))
                print(os.getpid(), count, 'SELECT results, count_accessed FROM results_passages WHERE query_hash = "{}";'.format(query_hash))
                rows = cur.fetchall()
                if len(rows) == 0:
                    count += 1
                    time.sleep(0.1)
                    continue
                else:
                    results = {**results, **json.loads(rows[0]['results'].decode('utf-8'))}
                    break

            count_accessed = 1

            # only load sections if there are no errors
            if 'errors' in results and len(results['errors']) > 0:
                results['sections'] = []
            else:
                results['sections'] = get_sections(tokens, doc_type_filters, collection_filters, availability_filters,
                       passage_length, passages_per_year, start_year, end_year, min_readability)



        cur.execute('UPDATE results_passages SET count_accessed = {} WHERE query_hash = "{}";'.format(count_accessed, query_hash))
        con.commit()
        con.close()

        print("Passages search for {} took {} and returned {} docs.".format(tokens, time.time() - start_time, len(results['sections'])))

        if token_parse_errors:
            if 'errors' in results:
                results['errors'] += token_parse_errors
            else:
                results['errors'] = token_parse_errors
        return results

    except:
        send_email('Text passages failure', '''Tokens: {}. Doc Type Filters: {}. Collection Filters: {}.
                            Availability Filters: {}.
                            Start Year: {}. End Year: {}. Passages Length: {}. Passages per Year: {}. Min Readability: {}.
                            Stack Trace: {}'''.format(
                            tokens, doc_type_filters, collection_filters, availability_filters,
                            start_year, end_year, passage_length, passages_per_year, min_readability,
                            traceback.format_exc()))


        try:
            con.close()
        except ProgrammingError:
            pass

        raise


def get_topic_model_results(tokens, doc_type_filters, collection_filters, availability_filters,
                              start_year, end_year, passage_length, passages_per_year, min_readability):
    """ Loads just the topic model that results from a query

    Topic models can take up to 1 minute to process and are loaded separately so there are already results to explore
    before the topic model is finished.

    :param tokens:
    :param doc_type_filters:
    :param collection_filters:
    :param availability_filters:
    :param start_year:
    :param end_year:
    :param passage_length:
    :param passages_per_year:
    :param min_readability:
    :return:
    """

    print("Getting text passages result", tokens, start_year, end_year)
    con = None


    tokens, errors, doc_type_filters, collection_filters, availability_filters = sanitize_query_input(
        tokens, doc_type_filters, collection_filters, availability_filters, type='passages'
    )

    query_hash = generate_hash((tokens, doc_type_filters, collection_filters, availability_filters, start_year, end_year,
                          passage_length, passages_per_year, min_readability))

    count = 0
    while True:
        if count == 2000:
            result = {'errors': "Something went wrong with this query and we terminated the calculation after 5 minutes."}
            send_email('Topic Model', '''Tokens: {}. Doc Type Filters: {}. Collection Filters: {}.
                        Availability Filters: {}.
                        Start Year: {}. End Year: {}. Passages Length: {}. Passages per Year: {}. Min Readability: {}.'''.format(
                        tokens, doc_type_filters, collection_filters, availability_filters,
                        start_year, end_year, passage_length, passages_per_year, min_readability))
            con.close()
            return result
        con, cur = DB.connect()
        cur.execute('SELECT results, count_accessed FROM results_passages WHERE query_hash = "{}"'.format(query_hash))
        print(os.getpid(), count, 'topic_model', 'SELECT results, count_accessed FROM results_passages WHERE query_hash = "{}"'.format(query_hash))
        rows = cur.fetchall()
        if len(rows) == 0:
            count += 1
            con.close()
            time.sleep(0.2)
            continue
        else:
            results = json.loads(rows[0]['results'].decode('utf-8'))
            if len(results['topic_model']) == 0:
                time.sleep(0.3)
                count += 1
                con.close()
                continue
            else:
                con.close()
                return results['topic_model']




def get_sections(tokens, doc_type_filters, collection_filters, availability_filters, passage_length,
                 passages_per_year, start_year, end_year, min_readability):
    """ Loads the sections for a text passages query

    :param tokens:
    :param doc_type_filters:
    :param collection_filters:
    :param availability_filters:
    :param passage_length:
    :param passages_per_year:
    :param start_year:
    :param end_year:
    :param min_readability:
    :return:
    """

    try:
        output_sections = []
        section_hash = generate_hash((tokens, doc_type_filters, collection_filters, availability_filters, passage_length))

        # load only correct passages if params == default
        if passage_length == 600 and passages_per_year == 100 and min_readability == 0.85:
            print('default true')
            con, cur = DB.connect()
            cur.execute('''SELECT results, year FROM results_passages_yearly_default
                                WHERE query_hash = "{}" AND year >= {} AND year <= {} ORDER BY year ASC;'''.format(section_hash, start_year, end_year))

            rows = cur.fetchall()
            print("len rows", len(rows), len(rows) == end_year+1 - start_year)
            if len(rows) == end_year+1 - start_year:
                for row in rows:
                    output_sections += json.loads(row['results'].decode('utf-8'))
                return output_sections

        random.seed(passages_per_year)
        shuffled_section_ids = list(range(5000))
        random.shuffle(shuffled_section_ids)
        year_counter = Counter()

        # make sure all years available
        # all years are available if count of rows == number of years to load
        cur = None  # make cur from while loop available below
        con = None

        count = 0
        while True:

            if count == 3000:
                print("did not find section result after 3000 loops or 5 minutes for {}".format(tokens, doc_type_filters, collection_filters, availability_filters))
                result = {'errors': "Something went wrong with this query and we terminated the calculation after 5 minutes."}
                send_email('Text passages failure', '''Tokens: {}. Doc Type Filters: {}. Collection Filters: {}.
                            Availability Filters: {}.
                            Start Year: {}. End Year: {}. Passages Length: {}. Passages per Year: {}. Min Readability: {}.'''.format(
                            tokens, doc_type_filters, collection_filters, availability_filters,
                            start_year, end_year, passage_length, passages_per_year, min_readability))
                con.close()
                return result

            con, cur = DB.connect()
            cur.execute('''SELECT COUNT(*) as count_years FROM results_passages_yearly
                                WHERE query_hash = "{}" AND year >= {} AND year <= {}'''.format(section_hash, start_year, end_year))
            print(os.getpid(), count, 'sections', '''SELECT COUNT(*) as count_years FROM results_passages_yearly
                                WHERE query_hash = "{}" AND year >= {} AND year <= {}'''.format(section_hash, start_year, end_year))
            count_years = cur.fetchall()[0]['count_years']
            if count_years == end_year+1 - start_year:
                break
            else:
                time.sleep(0.1)
                count += 1

        cur.execute('''SELECT results, year FROM results_passages_yearly
                          WHERE query_hash = "{}" AND year >= {} AND year <= {} ORDER BY year ASC;'''.format(section_hash, start_year, end_year))

        while True:
            sections_row = cur.fetchone()

            if not sections_row: break
            year = sections_row['year']
            year_sections = json.loads(sections_row['results'].decode('utf-8'))

            if year_sections == []: continue
            for section_id in shuffled_section_ids:
                if year_counter[year] == passages_per_year:
                    break

                # shuffled_section_ids goes to 4999 -> there will often be misses
                try:
                    section = year_sections[section_id]
                except IndexError:
                    continue

                if section[6] > min_readability:
                    output_sections.append(section)
                    year_counter[year] += 1

        con.close()
        return output_sections

    except:
        send_email('Section passages failure', '''Tokens: {}. Doc Type Filters: {}. Collection Filters: {}.
                            Availability Filters: {}.
                            Start Year: {}. End Year: {}. Passages Length: {}. Passages per Year: {}. Min Readability: {}.
                            Traceback: {}.'''.format(
                            tokens, doc_type_filters, collection_filters, availability_filters,
                            start_year, end_year, passage_length, passages_per_year, min_readability,
                            traceback.format_exc()))


        try:
            con.close()
        except ProgrammingError:
            pass

        raise


def sanitize_query_input(tokens, doc_type_filters, collection_filters, availability_filters, term_filters=None, type='frequencies'):
    """Sanitizes query input and returns all strings

    :param tokens:
    :param doc_type_filters:
    :param collection_filters:
    :param availability_filters:
    :param term_filters:
    :param type:
    :return:
    """

    error = ''
    process_despite_error = False

    if type == 'frequencies':
        tokens = sorted([token.strip().lower() for token in tokens.split(',') if token.strip() != ''])
    elif type == 'passages':
        tokens = [token.strip().lower() for token in tokens.split(',') if token.strip() != '']
        if len(tokens) > 1:
            tokens = [tokens[0]] + sorted(tokens[1:])

        if SANITIZE_REGEX.search(" ".join(tokens)):
            error = "Search terms can only contain letters, numbers, spaces, commas, and asterisks but not '{}'.\n".format(
                " ".join(set(SANITIZE_REGEX.findall(" ".join(tokens))))
            )

        tokens = [i.replace('-', ' ') for i in tokens]

        # replace quotes with commas so they get recognized as terms
        # "nicotine addiction" tar -> ['nicotine addiction', 'tar']
        for char in ['\'', '\"']:
            if " ".join(tokens).find(char) > -1:
                final_tokens = []
                for token in tokens:
                    token = token.replace(char, ',')
                    for ts in token.split(','):
                        if len(ts.strip()) > 0:
                            final_tokens.append(ts.strip())
                tokens = final_tokens
    else:
        raise ValueError("Only 'frequencies' and 'passages' are valid types for sanitize_tokens")


    doc_type_filters = sorted(list(eval(doc_type_filters)))

    # 8/9/18: collection filters can be ['msa_bat', 13], which is unorderable -> return unsorted in that case.
    try:
        collection_filters = sorted(list(eval(collection_filters)))
    except TypeError:
        collection_filters = list(eval(collection_filters))
    availability_filters = sorted(list(eval(availability_filters)))

    print("tokens after sanitize", tokens)

    if type == 'frequencies':
        term_filters = sorted([term_filter.strip().lower() for term_filter in term_filters.split(',') if term_filter.strip() != ''])
        return tokens, doc_type_filters, collection_filters, availability_filters, term_filters
    elif type == 'passages':
        return tokens, error, doc_type_filters, collection_filters, availability_filters


if __name__ == "__main__":

#    results = get_frequencies_results(tokens='addiction', doc_type_filters={'internal communication'},
#                                      collection_filters={}, availability_filters={}, term_filters=['nicotine'])
    # print(results)
#    r = get_text_passages_results(tokens='addic*, nicotine', doc_type_filters={}, collection_filters={}, availability_filters={},
#                                  start_year = 1970, end_year = 2000, passage_length=500, passages_per_year=500, min_readability=0.8)
    #print(r)

    q()
    pass

#    sanitize_tokens_and_frequencies('test', [], [], [], type='passages')

#    print(get_sections(['addiction'], ))

    # r = get_text_passages_results(tokens='addiction', doc_type_filters='[]', collection_filters='[]', availability_filters='[]',
    #                               start_year=1950, end_year=1960, passage_length=600, passages_per_year=1, min_readability=0.85)
    #
    # print(r)

