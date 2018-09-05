import random
from collections import defaultdict

import numpy as np
cimport numpy as np

def get_doc_ids_and_offsets(np.ndarray[np.uint8_t, ndim=1] token_vector, int year, list year_section_id_list,
                            np.ndarray[np.int32_t, ndim=2] section_to_doc_and_offset_arr):
    """ Creates a dict of lists with the document ids and section passages to process.

    Each key in doc_ids_and_offsets indicates a document id that we need to process
    Each dict key points to a list of tuples that indicate the starting and end point of a section to look at
    e.g. doc_ids_and_offsets[10430] = [(200, 500), (54666, 549660] -> two sections in document 10430 to process

    Note: the selection process is limited to 1000 documents per year and 100 passages per document.

    :param token_vector:
    :param year:
    :param year_section_id_list:
    :param section_to_doc_and_offset_arr:
    :return:
    """

    cdef int first_section_id = year_section_id_list[year-1901][0]
    cdef int last_section_id = year_section_id_list[year-1901][1]

    cdef int current_id, section_start, section_doc_id, cur_doc_id, doc_section_start, doc_section_end
    cdef int count_sections = 0

    # we check for id + 1. This avoids an out of bounds error
    if year == 2016:
        last_section_id -= 1

    doc_ids_and_offsets = defaultdict(list)

    current_id = first_section_id

    while True:

        if current_id == last_section_id + 1:
            break

        # if current_id is 0, we're currently not in a relevant passage
        if token_vector[current_id] == 0:
            current_id += 1

        # else, we have found a section
        else:
            # store the starting point
            section_start = current_id
            section_doc_id = section_to_doc_and_offset_arr[section_start, 0]

            while True:

                # look at next section
                current_id += 1
                if current_id == last_section_id + 1:
                    break

                else:
                    # if 1 and same document, the current section continues
                    cur_doc_id = section_to_doc_and_offset_arr[current_id, 0]
                    if token_vector[current_id] == 1 and section_doc_id == cur_doc_id:
                        continue

                    # if not, the section has ended at the last id and we need to store it
                    else:

                        doc_id = section_to_doc_and_offset_arr[section_start, 0]
                        doc_section_start = max(0, section_to_doc_and_offset_arr[section_start, 1] -1000)
                        doc_section_end = section_to_doc_and_offset_arr[current_id -1, 2] + 1000

                        # in rare cases, the exact same text passage appears multiple times in the same document
                        # which leads to doc_section_end < doc_section_start
                        # just skip those.
                        if doc_section_start < doc_section_end:
                            doc_ids_and_offsets[doc_id].append((doc_section_start, doc_section_end))
                            count_sections += 1
                        # no id increment. the increment will happen in the main while loop
                        break


    # limit to 1000 documents to process
    if len(doc_ids_and_offsets) > 1000:
        doc_ids_and_offsets_rnd = {}
        # sample requires a list, not a dict. so all the selected documents are added to a new dict.
        selected_ids = random.sample(list(doc_ids_and_offsets), 1000)
        for doc_id in selected_ids:
            doc_ids_and_offsets_rnd[doc_id] = doc_ids_and_offsets[doc_id]
        doc_ids_and_offsets = doc_ids_and_offsets_rnd

    # limit to 100 sections per document
    for doc_id in doc_ids_and_offsets:
        if len(doc_ids_and_offsets[doc_id]) > 100:
            doc_ids_and_offsets[doc_id] = random.sample(doc_ids_and_offsets[doc_id], 100)

    return doc_ids_and_offsets, count_sections
