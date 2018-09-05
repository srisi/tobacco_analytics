import re


SECTION_PATTERN = re.compile(r'\s.+\s', re.IGNORECASE | re.MULTILINE|re.DOTALL)


def get_final_sections(list barray_sections, bytearray first_token_barray, int passage_length, search_regexes):
    """ Extracts and returns the final sections from one document

    :param barray_sections: all the sections to be looked at in a document as byte arrays
    :param first_token_barray: byte array representation of the first token for fast searching
    :param passage_length: passage length search parameter
    :param search_regexes:
    :return:
    """

    cdef bytearray s

    cdef int start, hit, all_tokens_found, readability_sum
    cdef list section_tokens, final_sections

    cdef str section_raw, section, section_token

    cdef int passage_length_half = passage_length//2
    cdef int first_token_len = len(first_token_barray)

    final_sections = []

    for s in barray_sections:
        start = 0

        while True:

            hit = s.find(first_token_barray, start)
            if hit == -1:
                break

            all_tokens_found = 1
            section_raw = s[max(0, hit - passage_length_half): hit + passage_length_half].decode('utf-8',
                                                                                                 errors='ignore')

            for regex in search_regexes:
                if not regex.search(section_raw):
                    all_tokens_found = 0
                    break
            if all_tokens_found == 1:
                try:
                    section = SECTION_PATTERN.search(section_raw).group()
                    final_sections.append(section)
                except AttributeError:
                    print("Attribute error in text_passages_helper_process_doc. first_token: {}. regexes: {}. section_raw: {}.".format(first_token_barray, search_regexes, section_raw))

            start = hit + first_token_len

    return final_sections



def get_readability(str section, set vocabulary):
    """ Calculate the readability score (0.00 to 1.00) for a section

    :param section:
    :param vocabulary:
    :return:
    """

    cdef list section_tokens = section.split()
    cdef readability_sum = 0
    cdef str section_token

    for section_token in section_tokens:
        if section_token in vocabulary:
            readability_sum += 1
    readability = readability_sum / float(len(section_tokens))

    return readability


