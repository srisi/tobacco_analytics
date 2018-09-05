'''
Initializes full database in tob_full and fills tables for docs, doc_types, authors, and recipients.

2106-10-8: While re-running this script, received quite a few warnings saying:
Warning: (1366, "Incorrect string value: '\\xEF\\xBF\\xBDWil...' for column 'recipient' at row 1")
Maybe it's reading the table wrong somehow??

'''
import codecs
import datetime
import os
import re
import tarfile

from tobacco.configuration import VALID_COLLECTIONS, REMOVED_DOC_TYPES, PATH_OCR_FILES, DOC_COUNT
from tobacco.configuration import WORD_SPLIT_REGEX, SECTION_COUNT
from tobacco.frequencies_preprocessing.preprocessing_doc_types import get_dtype_dict
from tobacco.frequencies_preprocessing.preprocessing_docs import get_ocr_by_tid
from tobacco.frequencies_preprocessing.preprocessing_docs import get_tid_to_filelength_dict
from tobacco.utilities.databases import Database
from tobacco.utilities.ocr import expand_contractions
from tobacco.utilities.timestamp import calculate_unix_timestamp

IGNORED_DOC_TYPES = get_dtype_dict()['ignored']

print("timestamp of 1/1/1901: ", calculate_unix_timestamp('1901/1/1'))

def full_initialization_process():
    """ Run through the full database initialization process.

    This means:
    - It creates the TOBACCO_RAW DB ?unclear for what?
    - adds a unix timestamp to each document
    - Creates a utf8 text file for each document
    - Initializes doc, doc_types, authors, recipients tables
    - Fills those table

    1/19/2017
    Runs through the whole initialization after importing the raw sql
    Includes creating indexes
    :return:
    """

    db1 = Database('TOBACCO_RAW;')
    con1, cur1 = db1.connect()
    cur1.execute('create index idl_doc_field_id_idx on idl_doc_field(id);')
    cur1.execute('create index idl_doc_id_idx on idl_doc(id);')
    add_timestamp_to_idl_doc()

    create_utf_text_files()

    initialize_tables()
    fill_tables()


def add_timestamp_to_idl_doc():
    '''
    1/19/2017
    Before filling the tables, add a timestamp column to idl_doc, so we can order by timestamp when filling
    tob_full db.

    :return:
    '''

    db = Database("TOBACCO_RAW")
    con1, cur1 = db.connect()
    con2, cur2 = db.connect()

    #cur1.execute('ALTER TABLE idl_doc ADD timestamp BIGINT;')

    cur1.execute('SELECT id, value from idl_doc_field where itag = 4 order by id asc;')
    count = 0
    while True:
        count += 1
        if count % 10000 == 0:
            print(count)

        row = cur1.fetchone()
        if not row:
            break
        id = row['id']
        date = row['value']
        timestamp = calculate_unix_timestamp(date)

        if not type(timestamp) == int:
            print("Conversion error with: ".format(id, date, timestamp))

        if timestamp is None:
            cur2.execute('UPDATE idl_doc SET timestamp = NULL WHERE id={};'.format(id))
            print("No timestamp for", id, date, timestamp)
        else:
            cur2.execute('UPDATE idl_doc SET timestamp = {} WHERE id={};'.format(timestamp, id))
    con2.commit()

    cur1.execute('CREATE INDEX idl_doc_timestamp_idx on idl_doc(timestamp);')


def fill_tables():
    '''
    Fill the docs, doc_types, authors, and recipients tables
    Assigns each document an id, sorted by timestamp and TID, then collects and inserts all the necessary data

    :return:
    '''

    db1 = Database("TOBACCO_RAW")
    con1, cur1 = db1.connect()
    db2 = Database("TOBACCO_RAW")
    con2, cur2 = db2.connect()

    valid_tids = get_tid_to_filelength_dict()


    cur1.execute('''SELECT record_key, timestamp, collection_id, id as opt_min_id
                        FROM idl_doc
                        WHERE timestamp IS NOT NULL AND timestamp >= -2177452800 and industry_id=2
                        ORDER BY timestamp, record_key ASC;''')


    # Lists to store data until insert
    doc_types = []
    authors = []
    recipients = []
    docs = []

    idx = 0
    while True:


        # insert and reset after 10000 documents
        if len(docs) >= 10000:
            print("Current id: {}".format(idx))
            batch_insert(docs, doc_types, authors, recipients)
            docs = []
            doc_types = []
            authors = []
            recipients = []

        mand_doc = cur1.fetchone()
        if not mand_doc:
            break


        doc = {'id': idx,
               'tid': mand_doc['record_key'],
               'timestamp': mand_doc['timestamp'],
               # correct for utc to east coast time. (yes, this is an ugly hack but otherwise 1/1/1901 is interpreted as 12/31/1900
               'year': (datetime.datetime.fromtimestamp(mand_doc['timestamp']) + datetime.timedelta(hours=6)).year,
               'collection_id': mand_doc['collection_id'],
               'opt_min_id': mand_doc['opt_min_id'],
               'title': None,
               'pages': None,
        }

        doc_text = get_ocr_by_tid(doc['tid'], return_bytearray=False).lower()
        doc_text = expand_contractions(doc_text)
        doc['no_tokens'] = len(re.findall(WORD_SPLIT_REGEX, doc_text))

        tid = doc['tid']

        # 6/10/17 only add documents that actually contain text.
        if not tid in valid_tids:
            continue

        parsed_doc = parse_opt_min_doc(doc['opt_min_id'], doc['id'], cur2)

        # 1/31 if the document is from a doctype to remove, (e.g. trial list), then remove the document.
        if parsed_doc == 'remove doc':
            print ("Removing doc because trial list", doc)
            continue
        # skip non-tobacco collections
        if not mand_doc['collection_id'] in VALID_COLLECTIONS:
            continue


        doc.update(parsed_doc['doc'])


        try:
            if doc['date_orig'] == '1900':
                print("date == 1900", doc['date_orig'], doc)
                continue
        except KeyError:
            print("no date orig", doc)


        if doc['year'] < 1901:
            print ("Removing doc because < 1901", doc)
            continue
        else:


            docs.append(doc)
            doc_types += parsed_doc['doc_types']

            authors += parsed_doc['authors']
            recipients += parsed_doc['recipients']

            idx += 1

    batch_insert(docs, doc_types, authors, recipients)


def parse_opt_min_doc(opt_min_id, new_id, cur):
    '''
    Parses opt min doc and returns the data to the fill_tables function

    :param opt_min_doc:
    :return:
    '''

    doc_types = set()
    authors = set()
    recipients = set()


    cur.execute('SELECT id, itag, value from idl_doc_field where id = {} order by itag asc '.format(opt_min_id))
    opt_min_doc = cur.fetchall()

    doc = {}
    for line in opt_min_doc:
        if line['itag'] == 3:
            doc['title'] = line['value']
        elif line['itag'] == 4:
            doc['date_orig'] = line['value']

        # authors and recipients
        elif line['itag'] in [5,8]:
            authors.update(re.split('; ?', line['value']))
        elif line['itag'] in [49, 52]:
            recipients.update(re.split('; ?', line['value']))

        # doc types
        elif line['itag'] == 29:
            for dt in re.split('; ?', line['value']):
                # 1/31/2017
                if dt in REMOVED_DOC_TYPES:
                    return "remove doc"
                # 1/19/2017: only add doc types that are not ignored.
                if not dt in IGNORED_DOC_TYPES:
                    doc_types.add(dt)

        # availability
        elif line['itag'] == 32:
            doc['availability'] = line['value']+ ';'
        elif line['itag'] == 33:
            doc['availability'] += line['value']

        #pages
        elif line['itag'] == 100:
            doc['pages'] = line['value']


    doc_types = [{'doc_id': new_id,
                  'doc_type': doc_type.strip(), 'weight': 1/len(doc_types) } for doc_type in doc_types]
    authors = [{'doc_id': new_id,
                'author': author.strip()} for author in authors]
    recipients = [{'doc_id': new_id,
                   'recipient': recipient.strip()} for recipient in recipients]

    return {'doc': doc,
            'doc_types': doc_types,
            'authors': authors,
            'recipients': recipients}

def add_no_tokens():

    db = Database('TOB_FULL')
    con1, cur1 = db.connect()
    con2, cur2 = db.connect()

    cur1.execute("SELECT tid FROM docs;")

    while True:
        row = cur1.fetchone()
        if not row: break

        tid = row['tid']
        doc_text = get_ocr_by_tid(tid, return_bytearray=False).lower()
        doc_text = expand_contractions(doc_text)
        no_tokens = len(re.findall(WORD_SPLIT_REGEX, doc_text))

        print(tid, no_tokens)
        cur2.execute('UPDATE docs SET no_tokens = {} WHERE tid = "{}";'.format(no_tokens, tid))


    con2.commit()


def fill_in_unknown_authors():

    '''
    Add an empty string as author for all documents that have no authors
    Otherwise, the join in text_passages search does not work (joining docs and authors)

    :return:
    '''

    db = Database("TOB_FULL")
    con, cur = db.connect()

    for i in range(DOC_COUNT):
        cur.execute('select doc_id FROM authors WHERE doc_id = {}'.format(i))
        if len(cur.fetchall()) == 0:
            print(i, "inserting empty string")
            cur.execute('INSERT INTO authors VALUES ({}, "");'.format(i))

    con.commit()


def add_section_offset():
    '''

    This really adds the number of tokens per document.

    :return:
    '''

    db = Database("TOB_FULL")
    con1, cur1 = db.connect()
    cur1.execute("SELECT id, tid, no_tokens FROM docs ORDER BY id ASC;")

    count = 0
    first_section_id_of_doc = 0
    no_t = None
    doc_id = None
    while True:
        row = cur1.fetchone()
        if not row: break
        count += 1



        if count % 100000 == 0:
            print(count, first_section_id_of_doc)

        if count < 100:

            doc_text = get_ocr_by_tid(row['tid'], return_bytearray=False).lower()
            doc_text = expand_contractions(doc_text)
            document_split = re.findall(WORD_SPLIT_REGEX, doc_text)
            text_sections = [document_split[i:i+200] for i in range(0, len(document_split), 200)]

            print(count, first_section_id_of_doc, row['no_tokens'], row['no_tokens']//200+1, len(text_sections))


        first_section_id_of_doc = first_section_id_of_doc + row['no_tokens'] // 200 + 1

        # prevent off by 1 error
        if row['no_tokens'] % 200 == 0:
            first_section_id_of_doc -= 1

        no_t = row['no_tokens']
        doc_id = row['id']

    print("final", doc_id, first_section_id_of_doc, no_t)
    print(first_section_id_of_doc - SECTION_COUNT)


def batch_insert(docs, doc_types, authors, recipients):

    '''
    Inserts the documents in batches

    :param docs:
    :param doc_types:
    :param authors:
    :param recipients:
    :return:
    '''


    db2 = Database("TOB_FULL")
    db2.batch_insert('docs',
                    ['id', 'tid', 'timestamp', 'year', 'date_orig', 'title', 'collection_id', 'pages', 'no_docs', 'availability'],
                    docs)
    db2.batch_insert('doc_types', ['doc_id', 'doc_type', 'weight'], doc_types)
    db2.batch_insert('authors', ['doc_id', 'author'], authors)
    db2.batch_insert('recipients', ['doc_id', 'recipient'], recipients)


def create_utf_text_files():

    tar_folder = '/pcie/tobacco/'
    for filename in ['f-j.tar.gz', 'k-n.tar.gz', 'p-s.tar.gz', 't-z.tar.gz']:
        tar = tarfile.open(tar_folder + filename)

        count = 0
        for member in tar.getmembers():
            f=tar.extractfile(member)
            if not member.get_info()['name'].endswith('.ocr'):continue
            try:
                text = f.read().decode('cp1252', errors='ignore').lower()

                # 8/1/2017 why did I not add the contractions here before creating the folder initiallay????
                # 8/1/2017 added now
                text = expand_contractions(text)

                text = " ".join(WORD_SPLIT_REGEX.findall(text))
                tid = member.get_info()['name'][-12:-4]

                path = PATH_OCR_FILES + '{}/{}/{}/{}/'.format(tid[0], tid[1], tid[2], tid[3])
                if not os.path.exists(path):
                    os.makedirs(path)

                try:
                    file = codecs.open(path + tid + '.txt', "w", "utf-8")
                    file.write(text)
                    file.close()
                except FileNotFoundError:
                    print(member.get_info())

                count += 1
                if count % 10000 == 0:
                    print(filename, count)
            except AttributeError:
                pass
        tar.close()


def initialize_tables():

    db = Database("TOB_FULL")
    con, cur = db.connect()


#     cur.execute('''CREATE TABLE IF NOT EXISTS docs(
#                       id          integer       UNIQUE NOT NULL,
#                       tid         varchar(8)    UNIQUE NOT NULL,
#
#                       timestamp   BIGINT           NOT NULL,
#                       year        int           NOT NULL,
#                       date_orig   varchar(100)  NOT NULL,
#
#                       title       varchar(255)  ,
#                       collection_id  varchar(20)   NOT NULL,
#                       pages       int           ,
#
#                       availability  varchar(255)  NOT NULL,
#
#                       INDEX id_idx (id)
# )
#
#     ''')

    cur.execute('''CREATE TABLE IF NOT EXISTS docs(
                      id          integer       UNIQUE NOT NULL,
                      tid         varchar(8)    UNIQUE NOT NULL,

                      timestamp   BIGINT           NOT NULL,
                      year        int           NOT NULL,
                      date_orig   varchar(100)  NOT NULL,

                      title       varchar(255)  ,
                      collection_id  varchar(20)   NOT NULL,
                      pages       int           ,
                      no_tokens   int,

                      availability  varchar(255)  NOT NULL,

                      INDEX id_idx (id),
                      INDEX tid_idx (tid)
)

    ''')



    cur.execute('''CREATE TABLE IF NOT EXISTS doc_types(
                      doc_id        integer       NOT NULL,
                      doc_type      varchar(50)  NOT NULL,
                      weight        FLOAT         NOT NULL,


                      INDEX doc_id_idx (doc_id),
                      FOREIGN KEY (doc_id)
                        REFERENCES docs(id)
                        ON DELETE CASCADE,

                      CONSTRAINT unq_doc_id_type UNIQUE(doc_id, doc_type));
                      ''')



    # table to store authors
    cur.execute('''CREATE TABLE IF NOT EXISTS authors(
                      doc_id      integer       NOT NULL,
                      author      varchar(150)  NOT NULL,


                      INDEX doc_idx (doc_id),
                      FOREIGN KEY (doc_id)
                        REFERENCES docs(id)
                        ON DELETE CASCADE,

                      CONSTRAINT unq_doc_id_author UNIQUE(doc_id, author));
                      ''')


    # table to store recipients
    # if there are 3 recipients, each will have weight 1/3
    cur.execute('''CREATE TABLE IF NOT EXISTS recipients(
                      doc_id      integer        NOT NULL,
                      recipient  varchar(150)  NOT NULL,

                      INDEX id_idx (doc_id),
                      FOREIGN KEY (doc_id)
                        REFERENCES docs(id)
                        ON DELETE CASCADE,

                      CONSTRAINT unq_doc_id_recipient UNIQUE(doc_id, recipient));
                      ''')

    # table to store ngram vectors of tokens
    cur.execute('''CREATE TABLE IF NOT EXISTS tokens(
                      token     varchar(100)    NOT NULL,
                      token_reversed varchar(100) NOT NULL,
                      id        integer         NOT NULL,
                      ngram     integer         NOT NULL,
                      total     integer         NOT NULL,

                      INDEX token_reversed_idx (token_reversed),
                      CONSTRAINT unq_token_ngram UNIQUE(token, ngram));'''
                        )



if __name__ == '__main__':
#    initialize_tables()
#    fill_tables()
#    add_timestamp_to_idl_doc()
#    initialize_tables()

#    initialize_tables()
#    add_timestamp_to_idl_doc()
#    fill_tables()

#    add_no_tokens()
    add_section_offset()