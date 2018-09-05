import array
import pickle
import re
from collections import Counter

from tobacco.configuration import WORD_SPLIT_REGEX, SECTION_LENGTH, PATH_TOKENIZED
from tobacco.frequencies_preprocessing.preprocessing_docs import get_ocr_by_tid
from tobacco.full_db.s1_create_vocabulary import ngram_generator
from tobacco.utilities.databases import Database
from tobacco.utilities.ocr import load_vocabulary_trie, expand_contractions

def tokenize_document(tid, vocabulary, ngram, vocabulary_offset):
    '''
    July 2016

    Loads a tid text file and returns the individual tokens as an array

    '''


    text = get_ocr_by_tid(tid, return_bytearray=False)

    indices_counter = Counter()
    for token in ngram_generator(text, ngram):
        token = " ".join(token)
        try:
            token_id = vocabulary[token] - vocabulary_offset
            indices_counter[token_id] += 1
        except KeyError:
            pass

    indices = array.array(str("l"))
    data = array.array(str("l"))
    for index in sorted(indices_counter):
        indices.append(index)
        data.append(indices_counter[index])

    return (indices, data)

def tokenize_document_to_sections(tid, vocabulary, ngram, vocabulary_offset):
    '''

    Instead of tokenizing the whole document (tokenize_document()), this function tokenizes the document in sections of
    n words.

    :param tid:
    :param vocabulary:
    :param ngram:
    :param vocabulary_offset:
    :return:
    '''


    document = get_ocr_by_tid(tid, return_bytearray=False)
    document = document.lower()
    document = expand_contractions(document)
    document_split = re.findall(WORD_SPLIT_REGEX, document)

#    text_sections = " ".join([document_split[i:i+section_length] for i in range(0, len(document_split), section_length)])
    text_sections = [document_split[i:i+SECTION_LENGTH] for i in range(0, len(document_split), SECTION_LENGTH)]
    text_sections = [" ".join(text_section) for text_section in text_sections]

    indices_list = []
    data_list = []

    for text_section in text_sections:

        indices_counter = Counter()
        for token in ngram_generator(text_section, ngram):
            token = " ".join(token)
            try:
                token_id = vocabulary[token] - vocabulary_offset
                indices_counter[token_id] += 1

            except KeyError:
                pass

        indices = array.array(str("l"))
        data = array.array(str("l"))
        for index in sorted(indices_counter):
            indices.append(index)
            data.append(indices_counter[index])

        indices_list.append(indices)
        data_list.append(data)

    return (indices_list, data_list)


def yield_doc_ids():

    try:
        doc_ids = pickle.load(open(PATH_TOKENIZED + 'doc_ids.pickle', 'rb'))

    except IOError:
        db = Database("TOB_FULL")
        con, cur = db.connect()
        cur.execute("SELECT id, tid from docs order by id asc;")
        doc_ids = cur.fetchall()
        pickle.dump(doc_ids, open(PATH_TOKENIZED + "doc_ids.pickle", 'wb'), protocol=-1)


    print("loaded doc ids")

    for doc_id in doc_ids:
        yield doc_id

    print("finished yielding")


def test_tokenizer():
    '''
    10/7/16: written during move to python 3
    :return:
    '''

    'ffbb0005'
    vocabulary = load_vocabulary_trie(1)

    print(tokenize_document_to_sections('ffbb0005', vocabulary, 1, 0))


if __name__ == "__main__":
    test_tokenizer()