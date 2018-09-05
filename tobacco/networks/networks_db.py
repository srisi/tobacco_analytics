from tobacco.utilities.databases import Database
from tobacco.networks.networks_config import NETWORK_CONFIGS, SECTION_LENGTH
from tobacco.configuration import WORD_SPLIT_REGEX
from collections import Counter
import re

from tobacco.frequencies_preprocessing.preprocessing_docs import get_ocr_by_tid

english_name_regex = re.compile('[A-Z][a-zA-Z]+[, -]+[A-Z]{1,3}')

'''

Create a master filter instance
Use full tokenized section dtm as base
Then, filter for: node, year, token,
Return: dtm, col sums, row sums

'''


def get_nodes_init(main_name):
    '''
    Get initial set of 50 nodes to process

    :param main_name:
    :return:
    '''

    # Step 1: Identify top 50 nodes
    node_counter = Counter()

    db = Database("TOB_FULL")
    con, cur = db.connect()


    author_and_recipient_commands = [
       '''SELECT recipients.recipient as node, docs.tid, docs.year, "author" as main_person_is
                          FROM authors, recipients, docs
                          WHERE authors.author="{}" and authors.doc_id=recipients.doc_id AND docs.id=authors.doc_id
                          AND docs.year >= {} AND docs.year <= {};'''.format(
           NETWORK_CONFIGS[main_name]['name'], NETWORK_CONFIGS[main_name]['start_year'], NETWORK_CONFIGS[main_name]['end_year']),
       '''SELECT authors.author as node, authors.doc_id, docs.tid, docs.year, "recipient" as main_person_is
                          FROM authors, recipients, docs
                          WHERE recipients.recipient="{}" and authors.doc_id=recipients.doc_id AND docs.id=authors.doc_id
                          AND docs.year >= {} AND docs.year <= {};'''.format(
           NETWORK_CONFIGS[main_name]['name'], NETWORK_CONFIGS[main_name]['start_year'], NETWORK_CONFIGS[main_name]['end_year'])
    ]

    for command in author_and_recipient_commands:
        cur.execute(command)
        while True:
            row = cur.fetchone()
            if not row: break
            else:
                for person in english_name_regex.findall(row['node']):
                    node_counter[person] += 1

    print(node_counter.most_common(50))
    top_50_nodes = sorted([i[0] for i in node_counter.most_common(50)])



def generate_node_db(main_name):

    initialize_db(main_name)

    # Step 1: Identify top 50 nodes
    node_counter = Counter()

    db = Database("TOB_FULL")
    con, cur = db.connect()

    db_net = Database("TOB_NETWORKS")
    con_net, cur_net = db_net.connect()
    con.commit()

    author_and_recipient_commands = [
       '''SELECT recipients.recipient as node, docs.tid, docs.year, "author" as main_person_is
                          FROM authors, recipients, docs
                          WHERE authors.author="{}" and authors.doc_id=recipients.doc_id AND docs.id=authors.doc_id
                          AND docs.year >= {} AND docs.year <= {};'''.format(
           NETWORK_CONFIGS[main_name]['name'], NETWORK_CONFIGS[main_name]['start_year'], NETWORK_CONFIGS[main_name]['end_year']),
       '''SELECT authors.author as node, authors.doc_id, docs.tid, docs.year, "recipient" as main_person_is
                          FROM authors, recipients, docs
                          WHERE recipients.recipient="{}" and authors.doc_id=recipients.doc_id AND docs.id=authors.doc_id
                          AND docs.year >= {} AND docs.year <= {};'''.format(
           NETWORK_CONFIGS[main_name]['name'], NETWORK_CONFIGS[main_name]['start_year'], NETWORK_CONFIGS[main_name]['end_year'])
    ]

    for command in author_and_recipient_commands:
        cur.execute(command)
        while True:
            row = cur.fetchone()
            if not row: break
            else:
                for person in english_name_regex.findall(row['node']):
                    node_counter[person] += 1

    print(node_counter.most_common(50))
    top_50_nodes = [i[0] for i in node_counter.most_common(50)]

    # Step 2: insert
    for command in author_and_recipient_commands:
        cur.execute(command)
        while True:
            row = cur.fetchone()
            if not row: break
            else:
                for person in english_name_regex.findall(row['node']):
                    if person in top_50_nodes:
                        cur_net.execute('INSERT INTO {}_docs(node, main_author_is, tid, year) VALUES("{}", "{}", "{}", {})'.format(
                            main_name, person, row['main_person_is'], row['tid'], row['year']
                        ))

    con_net.commit()


def generate_section_table(main_name):

    db_net = Database("TOB_NETWORKS")
    con, cur = db_net.connect()
    con2, cur2 = db_net.connect()

    section_id = 0

    cur.execute('SELECT node, main_author_is, tid, weight, year FROM {}_docs ORDER BY id DESC;'.format(main_name))
    for row in cur.fetchall():
        document = get_ocr_by_tid(row['tid'], return_bytearray=False)
        document_split = re.findall(WORD_SPLIT_REGEX, document)
        text_sections = [document_split[i:i+SECTION_LENGTH] for i in range(0, len(document_split), SECTION_LENGTH)]
        text_sections = [" ".join(text_section) for text_section in text_sections]

        cur2.execute('SELECT COUNT(*) as count FROM {}_docs WHERE tid = "{}"'.format(main_name, row['tid']))
        weight = 1 / cur2.fetchall()[0]['count']

        for tid_section, section in enumerate(text_sections):
            cur2.execute('''INSERT INTO {}_sections (section_id, node, main_author_is, tid, tid_section, weight, year, text)
                                VALUES ({}, "{}", "{}", "{}", {}, {}, {}, "{}")'''.format(
              main_name, section_id, row['node'], row['main_author_is'], row['tid'], tid_section, weight, row['year'], section
            ))

            section_id += 1

    con2.commit()




def initialize_db(main_name):

    db = Database('TOB_NETWORKS')
    con, cur = db.connect()

    cur.execute('''CREATE TABLE IF NOT EXISTS {}_docs(
                    id              int NOT NULL AUTO_INCREMENT,
                    node            varchar(255)  NOT NULL,
                    main_author_is  varchar(10)   NOT NULL,
                    tid             varchar(10)   NOT NULL,
                    weight          float                 ,
                    year            INT           NOT NULL,

                    PRIMARY KEY(id));'''.format(main_name))


    cur.execute('''CREATE TABLE IF NOT EXISTS {}_sections(
                    section_id      int           NOT NULL,
                    node            varchar(255)  NOT NULL,
                    main_author_is  varchar(10)   NOT NULL,
                    tid             varchar(10)   NOT NULL,
                    tid_section     INT           NOT NULL,
                    weight          float         NOT NULL,
                    year            int           NOT NULL,
                    text            TEXT          NOT NULL,

                    PRIMARY KEY(section_id));'''.format(main_name))
    con.commit()

if __name__ == '__main__':

#    generate_node_db('dunn')
    initialize_db('dunn')
    generate_section_table('dunn')

