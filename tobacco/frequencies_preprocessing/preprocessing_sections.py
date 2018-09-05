import pickle
import re

import numpy as np
from tobacco.configuration import SECTION_COUNT, SECTION_LENGTH, PATH_TOKENIZED, WORD_SPLIT_REGEX
from tobacco.frequencies_preprocessing.preprocessing_docs import get_ocr_by_tid
from tobacco.utilities.databases import Database
from tobacco.utilities.ocr import expand_contractions


def get_doc_id_to_section_id_dict():
    """Returns a dict that contains maps from doc_id (0-indexed) to section ids.
    Every doc_id maps to a tuple of (first_section_id_of_doc, last_section_id_of_doc)

    :return:
    """

    try:
        doc_id_to_section_id_dict = pickle.load(open(PATH_TOKENIZED + 'doc_id_to_section_id_dict.pickle', 'rb'))
    except IOError:

        print("doc_id_to_section_id_dict not found. Creating a new one with section length = {}.".format(
            SECTION_LENGTH))

        db = Database("TOB_FULL")
        con1, cur1 = db.connect()
        cur1.execute("SELECT id, tid, no_tokens FROM docs ORDER BY id ASC;")

        doc_id_to_section_id_dict = {}

        first_section_id_of_doc = 0
        while True:
            row = cur1.fetchone()
            if not row: break

            last_section_id_of_doc = first_section_id_of_doc + row['no_tokens'] // 200

            # prevent off by 1 error
            if row['no_tokens'] % 200 == 0:
                last_section_id_of_doc -= 1

            doc_id_to_section_id_dict[row['id']] = (first_section_id_of_doc, last_section_id_of_doc)

            if row['id'] < 20 or row['id'] % 1000000 == 0:
                print(row['id'], first_section_id_of_doc, last_section_id_of_doc)

            first_section_id_of_doc = last_section_id_of_doc + 1

        print("Final section id: {}. Number of sections: {}. These numbers have to be equal.".format(
            first_section_id_of_doc, SECTION_COUNT))
        assert first_section_id_of_doc == SECTION_COUNT

        pickle.dump(doc_id_to_section_id_dict, open(PATH_TOKENIZED + 'doc_id_to_section_id_dict.pickle', 'wb'))

    return doc_id_to_section_id_dict


def get_section_to_doc_and_offset_arr():
    """" Returns a dict that maps from docs to sections and offsets

    :return:
    """

    try:
        section_to_doc_and_offset_arr = np.load(PATH_TOKENIZED + 'section_to_doc_and_offset_arr.npy')
    except IOError:

        print("section_to_doc_and_offset_dict not found. Creating a new one with section length = {}.".format(
            SECTION_LENGTH))

        db = Database("TOB_FULL")
        con1, cur1 = db.connect()
        cur1.execute("SELECT id, tid, no_tokens FROM docs ORDER BY id ASC;")

        section_to_doc_and_offset_arr = np.zeros((SECTION_COUNT, 3), dtype=np.int32)
        doc_id_to_section_id_dict = get_doc_id_to_section_id_dict()

        characters_to_delete_list = ['¯']

        while True:
            row = cur1.fetchone()
            if not row: break

            doc_id = row['id']
            first_section_id_of_doc = doc_id_to_section_id_dict[row['id']][0]

            if doc_id % 10000 == 0:
                print(doc_id)

            # load doc and get offsets
            document = get_ocr_by_tid(row['tid'], return_bytearray=False)
            document = expand_contractions(document)
            doc_len_orig = len(document)
            for character in characters_to_delete_list:
                document = document.replace(character, '')

            document_split = re.findall(WORD_SPLIT_REGEX, document)
            text_sections = [document_split[i:i+SECTION_LENGTH] for i in range(0, len(document_split), SECTION_LENGTH)]
            text_sections = [" ".join(text_section) for text_section in text_sections]

            no_deleted_characters = doc_len_orig - len(" ".join(document_split))
            # if no_deleted_characters > 0:
            #     print( doc_id, row['tid'], no_deleted_characters, doc_len_orig)

            for section_idx, section in enumerate(text_sections):
                offset = document.find(section)
                if offset == -1:
                    if offset == -1:
                        offset = 0
                    print("\nSection not found in ", row['tid'], doc_id, section_idx)
                    print(document[:200])
                    print(section[:200])
                section_to_doc_and_offset_arr[first_section_id_of_doc+section_idx][0] = doc_id
                # offset start
                section_to_doc_and_offset_arr[first_section_id_of_doc+section_idx][1] = offset
                # offset end (we deleted characters, which could move the end of the section no_deleted_charaters
                # towards the end. the third row entry stores that value
                section_to_doc_and_offset_arr[first_section_id_of_doc+section_idx][2] = offset + len(section) + no_deleted_characters

        np.save(PATH_TOKENIZED + 'section_to_doc_and_offset_arr.npy', section_to_doc_and_offset_arr)

    return section_to_doc_and_offset_arr



def get_offsets(tid):
    """

    :param tid:
    :return:
    """

    characters_to_delete_list = ['¯']

    document = get_ocr_by_tid(tid, return_bytearray=False)
    document = expand_contractions(document)
    for character in characters_to_delete_list:
        document = document.replace(character, '')

    document_split = re.findall(WORD_SPLIT_REGEX, document)
    text_sections = [document_split[i:i+SECTION_LENGTH] for i in range(0, len(document_split), SECTION_LENGTH)]
    text_sections = [" ".join(text_section) for text_section in text_sections]

    no_deleted_characters = len(document) - len(" ".join(document_split))

    offsets = []
    for section in text_sections:
        offsets.append(document.find(section))
        print("\n")
        print(document.find(section))
        print(document[:2000])
        print(section[:2000])
        print(no_deleted_characters)

if __name__ == "__main__":

    import time
    s = time.time()
    get_section_to_doc_and_offset_arr()
    print(time.time() - s)

    # get_offsets('qhgl0223')