
import datetime
import multiprocessing
import random

from tobacco.text_passages.text_passages_helper_doc_ids import get_doc_ids_and_offsets
from tobacco.text_passages.text_passages_helper_db import document_iterator, insert_passages_yearly_result
from tobacco.text_passages.text_passages_helper_process_doc import get_final_sections, get_readability
from tobacco.frequencies_preprocessing.preprocessing_docs import get_ocr_sections

def process_year_of_sections_cython(str first_token, list tokens, list search_regexes, token_vector, int year,
                                    int passage_length, active_filters, set vocabulary, globals,
                                    insert_result_to_db=True):
    """ Processes one year of a text passage search and returns them.

    """


    cdef bytearray first_token_encoded = bytearray(first_token.encode('utf-8'))

    cdef list doc_ids, raw_sections

    cdef str doc_id, date_str, collection, tid, title, author, general, token

    cdef int total_sections, complete

    cdef list tokens_with_removed_wildcards = [token.replace('*', '') for token in tokens]

    doc_ids_and_offsets, total_sections = get_doc_ids_and_offsets(token_vector, year,
                                                      globals['year_parts_id_list']['sections'],
                                                      globals['section_to_doc_and_offset_arr'])


    if total_sections < 2000:
        complete = 1
    else:
        complete = 0

    # if the year has no documents, insert empty list and return
    if len(doc_ids_and_offsets) == 0:
        insert_passages_yearly_result(tokens, active_filters, year, passage_length, complete, [])
        return

    doc_ids = list(doc_ids_and_offsets.keys())

    output_sections = []

    for row in document_iterator(doc_ids):

        date = datetime.datetime.fromtimestamp(row['timestamp']) + datetime.timedelta(hours=6)
        date_str = date.strftime('%Y-%m-%d')
        collection = globals['collections_and_idx_dict'][int(row['collection_id'])]['name_short']
        tid = row['tid']
        title = row['title']
        if title is None:
            title = ''
        if len(title) > 50:
            title = title[:50] + '...'
        author = row['author']
        if len(author) > 50:
            author = author[:50] + '...'
        general = '{}<br><br><a href="https://www.industrydocumentslibrary.ucsf.edu/tobacco/docs/#id={}" target="_blank">{}</a><br><br>{}'.format(date_str, tid, tid, collection)

        raw_sections = get_ocr_sections(tid, doc_ids_and_offsets[row['id']])

        final_sections = get_final_sections(raw_sections, first_token_encoded, passage_length, search_regexes)
        for final_section in final_sections:
            for token in tokens_with_removed_wildcards:
                final_section = final_section.replace(token, '<b>{}</b>'.format(token))

            output_sections.append((
                tid,
                title,
                year,
                author,
                date_str,
                collection,
                get_readability(final_section, vocabulary),
                final_section,
                general
            ))

    if len(output_sections) > 5000:
        output_sections = random.sample(output_sections, 5000)

    if insert_result_to_db:
        p = multiprocessing.Process(target=insert_passages_yearly_result, args=(tokens, active_filters, year, passage_length, complete, output_sections))
        p.start()

    return output_sections

#    insert_passages_yearly_result(tokens, active_filters, year, passage_length, complete, output_sections)
