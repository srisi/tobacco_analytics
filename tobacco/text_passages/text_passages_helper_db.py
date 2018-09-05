import json
import random

from tobacco.utilities.databases import Database
from tobacco.configuration import RESULTS_DB_NAME
from tobacco.utilities.hash import generate_hash

DB_TOB_RESULTS = Database(RESULTS_DB_NAME)

# passages per year for default settings remains 100
random.seed(100)
# maintain a pre-shuffled list of section ids to draw from
SHUFFLED_SECTION_IDS = list(range(5000))
random.shuffle(SHUFFLED_SECTION_IDS)

def document_iterator(doc_ids):
    """ Select metadata (authors, tid, date, title, collection) for selected text passages

    6/9/17: Now using bytearrays instead of mysql
    :param doc_ids:
    :return:
    """

    db = Database("TOB_FULL")
    con, cur = db.connect()


    cur.execute('''SELECT docs.id, docs.tid, docs.timestamp, docs.year, docs.title, docs.collection_id,
                          GROUP_CONCAT(authors.author SEPARATOR ', ') as author
                      FROM docs, authors
                      WHERE docs.id = authors.doc_id AND docs.id in ({}) GROUP BY docs.id;'''.format(
            ",".join([str(i) for i in doc_ids])))

    while True:
        row = cur.fetchone()
        if row:
            yield row
        else:
            break
    con.close()


def insert_passages_yearly_result(tokens, active_filters, year, passage_length, complete, output_sections):
    """ Insert the results of one year inot the results db

    The process runs through 2 storage options.
    First, it stores all results (up to 5000) in the database. Out of these, a requested result of, e.g. 400 can be
    selected later.
    Second, for faster loading, it stores a set with default search params (100 passages per year, 0.85 readability)
    in the results_passages_yearly_default table.

    :param tokens:
    :param active_filters:
    :param year:
    :param passage_length:
    :param complete:
    :param output_sections:
    :return:
    """

    con, cur = DB_TOB_RESULTS.connect()

    print("inserting yearly", year, len(output_sections))


    section_hash = generate_hash((tokens, active_filters['doc_type'], active_filters['collection'],
                                  active_filters['availability'], passage_length))

    store_cmd = '''REPLACE INTO results_passages_yearly (tokens,
                                                        doc_type_filters,
                                                        collection_filters,
                                                        availability_filters,
                                                        year,
                                                        passage_length,
                                                        complete,
                                                        results,
                                                        query_hash
                                                        )
                                              VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s);'''


    cur.execute(store_cmd, (str(tokens), str(active_filters['doc_type']), str(active_filters['collection']),
                            str(active_filters['availability']), year, passage_length, complete,
                            json.dumps(output_sections), section_hash))
    con.commit()


    default_sections = []
    year_counter = 0
    for section_id in SHUFFLED_SECTION_IDS:
        if year_counter  == 100:
            break
        try:
            section = output_sections[section_id]
        except IndexError:
            continue

        if section[6] > 0.85:
            default_sections.append(section)
            year_counter += 1


    store_cmd = '''REPLACE INTO results_passages_yearly_default (tokens,
                                                        doc_type_filters,
                                                        collection_filters,
                                                        availability_filters,
                                                        year,
                                                        results,
                                                        query_hash
                                                        )
                                              VALUES(%s, %s, %s, %s, %s, %s, %s);'''

    cur.execute(store_cmd, (str(tokens), str(active_filters['doc_type']), str(active_filters['collection']),
                            str(active_filters['availability']), year, json.dumps(default_sections), section_hash))
    con.commit()
    con.close()
