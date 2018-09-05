import datetime
import re
from collections import defaultdict, namedtuple, Counter
from tobacco.frequencies_preprocessing.preprocessing_collections import get_collection_preprocessing
from tobacco.utilities.databases import Database
from tobacco.utilities.ocr import load_vocabulary_trie
from tobacco.frequencies_preprocessing.preprocessing_docs import get_ocr_by_tid

import random


COLLECTION_NAME_TO_IDX_DICT = get_collection_preprocessing()[0]

Document = namedtuple('Document', ['tid', 'title', 'date', 'year', 'collection'])
Passage = namedtuple('Passage', ['Document', 'text'])

import numpy as np
import time
cimport numpy as np

DTYPE_FLOAT64 = np.float64
ctypedef np.float64_t DTYPE_FLOAT64_t
DTYPE_INT = np.int
ctypedef np.int_t DTYPE_INT_t



# def get_doc_ids_and_offsets(np.ndarray[np.uint8_t, ndim=1] token_vector, int year, list year_section_id_list,
#                             np.ndarray[np.int32_t, ndim=2] section_to_doc_and_offset_arr):
#
#     cdef int first_section_id = year_section_id_list[year-1901][0]
#     cdef int last_section_id = year_section_id_list[year-1901][1]
#
#     cdef int current_id, section_start, section_doc_id, cur_doc_id, doc_section_start, doc_section_end
#
#     cdef int count_sections
#
#     # we check for id + 1. This avoids an out of bounds error
#     if year == 2016:
#         last_section_id -= 1
#
#     doc_ids_and_offsets = defaultdict(list)
#     doc_ids_and_offsets_list = []
#
#     current_id = first_section_id
#
#     while True:
#
#         if current_id == last_section_id + 1:
#             break
#
#         # if current_id is 0, we're currently not in a relevant passage
#         if token_vector[current_id] == 0:
#             current_id += 1
#
#         # else, we have found a section
#         else:
#             # store the starting point
#             section_start = current_id
#             section_doc_id = section_to_doc_and_offset_arr[section_start, 0]
#
#             while True:
#
#                 # look at next section
#                 current_id += 1
#                 if current_id == last_section_id + 1:
#                     break
#
#                 else:
#                     # if 1 and same document, the current section continues
#                     cur_doc_id = section_to_doc_and_offset_arr[current_id, 0]
#                     if token_vector[current_id] == 1 and section_doc_id == cur_doc_id:
#                         continue
#
#                     # if not, the section has ended at the last id and we need to store it
#                     else:
#                       doc_id = section_to_doc_and_offset_arr[section_start, 0]
#                       doc_section_start = section_to_doc_and_offset_arr[section_start, 1]
#                       doc_section_end = section_to_doc_and_offset_arr[current_id -1, 2]
#
#                       doc_ids_and_offsets[doc_id].append((doc_section_start, doc_section_end))
#                       doc_ids_and_offsets_list.append((doc_id, doc_section_start, doc_section_end))
#                       count_sections += 1
#
#                       # # assert that the doc_id remains the same
#                       # assert section_to_doc_and_offset_arr[section_start][0] == section_to_doc_and_offset_arr[current_id-1][0]
#                       # no id increment. the increment will happen in the main while loop
#                       break
#
#
#     if count_sections > 2000:
#         doc_ids_and_offsets = random.sample(doc_ids_and_offsets, 2000)
#
#     docs_to_sections_dict = defaultdict(list)
#     for section in doc_ids_and_offsets:
#         # instead of using the previous section, just extend the section slightly in both directions
#         start = min(0, section[0] - 1000)
#         end = section[1] + 1000
#         docs_to_sections_dict[section[0]].append((start, end))
#
#     return docs_to_sections_dict, count_sections



# def select_doc_ids_np(np.ndarray[np.uint8_t, ndim=1] token_vector, int start_year, int end_year, int docs_per_year, year_section_id_list, return_all=False):
#     '''
#     Randomly selects up to docs_per_year doc_ids in which the selected token appears
#
#
#     :param token_vector:
#     :param start_year:
#     :param end_year:
#     :param docs_per_year:
#     :param year_doc_id_list:
#     :return:
#     '''
#
#
#     # current year to process
#     cdef int current_year = start_year
#
#
#     # current idx in the indices array to process
#     cdef int indices_idx = 0
#
#     # start and end ids of the current year
#     cdef int current_year_start_id = year_section_id_list[current_year-1]
#     cdef int current_year_end_id = year_section_id_list[current_year]
#
#     # dict of years to list of doc_ids. Every year is a list of doc_ids wherein the token appears
#     doc_ids_by_year = {}
#     for year in range(start_year, end_year +1):
#         doc_ids_by_year[year] = []
#
#
#     cdef int first_id = year_section_id_list[start_year - 1]
#     cdef int final_id = year_section_id_list[end_year]
#
#     # pass through all nonzero indices
#     s = time.time()
#     cdef np.ndarray[long, ndim=1] vector_indices = token_vector[first_id: final_id].nonzero()[0]
#
#     cdef int vector_indices_len = len(vector_indices)
#     cdef int vector_index = 0
#     cdef int section_id
#
#
#     while True:
#
#         # first_id is an offset. np.nonzero starts at this offset
#         section_id = vector_indices[vector_index] + first_id
#
#         # if doc_id is > than the highest id, stop loop
#         if section_id >= final_id:
#             break
#
#
#         elif section_id < current_year_start_id:
#             vector_index += 1
#             #break if all indices are processed
#             if vector_index >= (vector_indices_len):
#                 break
#
#         # if the doc_id is between start and end id, add it to the list
#         elif section_id >= current_year_start_id and section_id < current_year_end_id:
#             doc_ids_by_year[current_year].append(section_id)
#             vector_index += 1
#
# #            if current_year == 1950 or current_year == 1951: print(doc_id)
#
#             # if all indices are processed, break.
#             if vector_index == (vector_indices_len):
#                 break
#
#         # if the doc id is beyond the current year, move to the next year
#         elif section_id >= current_year_end_id:
#             current_year += 1
#             current_year_start_id = year_section_id_list[current_year - 1]
#             current_year_end_id = year_section_id_list[current_year]
#         else:
#             print("Encountered unexpected state. Please debug me.")
#
#     # finally, add all doc_ids to a unified list.
#     doc_ids = []
#     for year in doc_ids_by_year:
#         if len(doc_ids_by_year[year]) < int(1.2 * docs_per_year) or return_all:
#             doc_ids += doc_ids_by_year[year]
#         else:
# #            doc_ids_by_year[year] = random.sample(doc_ids_by_year[year], docs_per_year)
#             doc_ids += random.sample(doc_ids_by_year[year], int(1.2 * docs_per_year))
#
#     return doc_ids


SECTION_PATTERN = re.compile(r'\s.+\s', re.IGNORECASE | re.MULTILINE|re.DOTALL)
cdef set VOCABULARY = load_vocabulary_trie(1, return_type='set')
VOCABULARY_TRIE = load_vocabulary_trie(1, return_type='trie')
WORD_SPLIT_REGEX = re.compile(r"\b\w+\b")

def get_docs_to_process_cython(list doc_ids, str first_token, list search_regexes, int passage_length, float min_readability):
    '''
    Finds all docs to process for a given year

    :param doc_ids:
    :param passages_per_year:
    :return:
    '''

    cdef int year, start, hit, readability_sum
    cdef int len_first_token = len(first_token)
    cdef str date_str, collection, section_raw, section, section_token
    cdef list section_tokens
    cdef bytearray barray, first_token_encoded


    text_passages_by_year = defaultdict(list)
    text_passages_totals = np.zeros(len(VOCABULARY), dtype=np.int64)
    author_counter = Counter()

    first_token_encoded = bytearray(first_token.encode('utf-8'))

    for row in document_iterator(doc_ids):

        date = datetime.datetime.fromtimestamp(row['timestamp']) + datetime.timedelta(hours=6)
        year = row['year']

        date_str = date.strftime('%Y-%m-%d')
        collection = COLLECTION_NAME_TO_IDX_DICT[int(row['collection_id'])]['name_short']
        barray = row['barray']

        start = 0
        while True:
            hit = barray.find(first_token_encoded, start)
            if hit == -1:
                break

            all_tokens_found = True
            section_raw = barray[max(0, hit - passage_length//2): hit + passage_length//2].decode('utf-8', errors='ignore')

            for regex in search_regexes[0:]:
                if not regex.search(section_raw):
                    all_tokens_found = False
                    break
            if all_tokens_found:
                section = SECTION_PATTERN.search(section_raw).group()
                section_tokens = section.split()

                # this used to be a list comp but splitting it makes it 10 times faster
                readability_sum = 0
                for section_token in section_tokens:
                    if section_token in VOCABULARY:
                        readability_sum += 1

                        # 7/18: adding distinctive tokens
                        text_passages_totals[VOCABULARY_TRIE[section_token]] += 1

                readability = readability_sum / float(len(section_tokens))

                if readability >= min_readability:
                    section = " ".join(section_tokens)
                    author_counter[row['author']] += 1
                    text_passages_by_year[year].append({'tid': row['tid'],
                                                        'title': row['title'],
                                                        'year': row['year'],
                                                        'author': row['author'],
                                                        'date': date_str,
                                                        'collection': collection,
                                                        'text': section})

            start = hit + len_first_token

    return text_passages_by_year, text_passages_totals

def document_iterator(doc_ids):
    '''
    6/9/17: Now using bytearrays instead of mysql
    :param doc_ids:
    :return:
    '''

    db = Database("TOB_FULL")
    con, cur = db.connect()

#    cur.execute('''SELECT id, tid, timestamp, year, title, collection_id FROM docs WHERE id in ({});'''.format(
#            ",".join([str(i) for i in doc_ids])))

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










































