import math

import numpy as np
cimport numpy as np
from tobacco.utilities.ocr import get_vocabulary_totals, load_vocabulary_trie

from tobacco.configuration import STOP_WORDS_SKLEARN, YEAR_COUNT

DTYPE_FLOAT64 = np.float64
ctypedef np.float64_t DTYPE_FLOAT64_t


VOCABULARY_TRIE = load_vocabulary_trie(1)
GLOBAL_TOTALS = get_vocabulary_totals(1)
GLOBAL_TOTALS_SUM = GLOBAL_TOTALS.sum()

STOP_WORDS = set(STOP_WORDS_SKLEARN).union(
        {'pgnbr', 'quot', 'amp', 'apos', '0', '1','2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
         '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
         'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
         'w', 'x', 'y', 'z'}).union(set(str(i) for i in range(0, 1800)))



def calculate_distinctive_terms(np.ndarray[long, ndim=1] passage_totals, list output_sections, search_tokens):
    """ Calculates distinctive terms for the list of sections.

    :param passage_totals: np array of the 1 gram counts for all words in the vocabulary
    :param output_sections: list of all selected text passages
    :param search_tokens: The original search tokens
    :return:
    """

    global VOCABULARY_TRIE
    global GLOBAL_TOTALS
    global GLOBAL_TOTALS_SUM

    # can't declare types for global vars, so it's done here
    cdef np.ndarray[long, ndim=1] glob_totals = GLOBAL_TOTALS
    cdef long glob_totals_sum = GLOBAL_TOTALS_SUM

    cdef np.ndarray[double, ndim=1] log_likelihoods = np.zeros(len(passage_totals))

    cdef double passage_totals_sum, a, b, e1, e2, a_plus_b_div_by_totals
    cdef int i, token_id

    cdef str section_text, token

    cdef set tokens_already_found_in_section

    # null out the original search tokens in passage_totals (but don't fuck with the global... this is not a copy)
    # idea: we don't want the search tokens to show up as distinctive terms
    for token in search_tokens:
        try:
            token_id = VOCABULARY_TRIE[token]
            passage_totals[token_id] = 0
        # 2 grams don't appear in the vocabulary
        except KeyError: pass
        if len(token.split()) >= 2:
            for token_split in token.split():
                try:
                    token_id = VOCABULARY_TRIE[token]
                    passage_totals[token_id] = 0
                except KeyError: pass


    passage_totals_sum = passage_totals.sum()

    for i in range(len(passage_totals)):
        a = passage_totals[i]
        b = glob_totals[i]
        if a == 0 or b == 0: continue

        a_plus_b_div_by_totals = (a+b) / (passage_totals_sum+glob_totals_sum)

        e1 = passage_totals_sum * a_plus_b_div_by_totals
        e2 = glob_totals_sum * a_plus_b_div_by_totals

        g2 = 2 * ((a * math.log(a / e1)) + b * math.log(b / e2))

        if a/e1 < 1: # equivalent to a * math.log(a/e1) < 0 (log function is 0 at 1)
            g2 = -g2

        log_likelihoods[i] = g2

    # sorting comes from https://stackoverflow.com/questions/6910641/how-to-get-indices-of-n-maximum-values-in-a-numpy-array
    max_indices = np.argpartition(log_likelihoods, -50)[-50:]
    max_indices = max_indices[np.argsort(log_likelihoods[max_indices])][::-1]

    distinctive_terms = []
    for i in max_indices:
        term = VOCABULARY_TRIE.restore_key(i)
        if term in STOP_WORDS_SKLEARN:
            continue
        distinctive_terms.append((term, passage_totals[i], round(log_likelihoods[i], 2), YEAR_COUNT * [0]))


    distinctive_terms = distinctive_terms[:20]

    # map from term to id
    distinctive_terms_to_id_dict = {t[0]: idx for idx, t in enumerate(distinctive_terms)}

    # fill list of sections per year
    for section in output_sections:
        # each token should only be added once per document (so they can be displayed later)
        # this set keeps track of tokens already added.
        tokens_already_found_in_section = set()
        for token in section[7].split():
            if token in distinctive_terms_to_id_dict and not token in tokens_already_found_in_section:
                token_id = distinctive_terms_to_id_dict[token]
                year = section[2]
                distinctive_terms[token_id][3][year-1901] += 1
                tokens_already_found_in_section.add(token)

    return distinctive_terms, log_likelihoods



def get_text_passages_totals(list output_sections, list search_tokens):
    """ Creates and returns an np array of the counts of all 1 grams as well as a set of the 2000 most frequent terms

    :param output_sections:
    :param tokens:
    :return:
    """

    global VOCABULARY_TRIE

    cdef set top_2000_terms = set()
    cdef int i
    cdef np.ndarray[long, ndim=1] passage_totals = np.zeros(len(VOCABULARY_TRIE), dtype=np.int64)
    cdef set stop_words
    cdef str section_text, token

    # create a set of all 1-grams in tokens
    token_set = set()
    for token in search_tokens:
        for one_gram in token.split():
            token_set.add(one_gram)
    stop_words = STOP_WORDS.union(token_set)


    for section in output_sections:
        for token in section[7].split():
            if token in VOCABULARY_TRIE:
                passage_totals[VOCABULARY_TRIE[token]] += 1


    for i in passage_totals.argsort()[-2000:][::-1]:
        token = VOCABULARY_TRIE.restore_key(i)
        if token in stop_words or passage_totals[i] < 5:
            continue
        else:
            top_2000_terms.add(token)

    return passage_totals, top_2000_terms


if __name__ == "__main__":
    calculate_distinctive_terms(None, None)