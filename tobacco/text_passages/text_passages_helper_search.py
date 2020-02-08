import re
import time
import numpy as np

from tobacco.frequencies_preprocessing.preprocessing_search import wildcard_search
#from tobacco.frequencies_preprocessing.preprocessing_tokens import get_ngram_vector
#from tobacco.utilities.sparse_matrices import csc_to_np_uint8

from tobacco.utilities.vector import Vector


def parse_text_passages_tokens(tokens):


    search_regexes = []

    intersection_vector = None

    first_token = tokens[0].strip('*')

    for token in tokens:
        # if wildcard, process wildcard
        if token.find('*') > -1:
            token_vector, token_regex = process_wildcard_token(token)

        # else: handle normal token
        else:
            try:

                token_vector = Vector().load_token_vector(token, return_type='np_uint8',
                                                          docs_or_sections='sections')
                token_regex = re.compile(r'\b{}\b'.format(token))

            # will throw a fof error when the token does not exist. e.g. 'compound w'
            except FileNotFoundError:
                token_vector, token_regex = process_nonexistant_token(token)

        if token_vector is not None:
            if intersection_vector is None:
                intersection_vector = token_vector
            else:
                intersection_vector *= token_vector


        search_regexes.append(token_regex)

    return first_token, search_regexes, intersection_vector

def process_nonexistant_token(token):

    token_vector = None
    token_regex = re.compile(r'\b{}\b'.format(token))
    for token_part in token.split():
        try:
            part_vector = get_ngram_vector(token_part, return_type='uint8', docs_or_sections='sections')
            print("non existent token." , token, "found: ", token_part, np.sum(part_vector))
            if token_vector is None:
                token_vector = part_vector
            else:
                token_vector *= part_vector
        except FileNotFoundError:
            continue


    return token_vector, token_regex



def process_wildcard_token(token):

    wildcard_vector = None
    matches = wildcard_search(token, ngram=len(token.split()), number_of_tokens=10)
    print(token, "no matches", len(matches), matches)
    if matches:
        for match in matches:
            if wildcard_vector is None:
                wildcard_vector = get_ngram_vector(match, return_type='csc', docs_or_sections='sections')
            else:
                wildcard_vector += get_ngram_vector(match, return_type='csc', docs_or_sections='sections')


        wildcard_vector = csc_to_np_uint8(wildcard_vector)

    if token[0] == '*':
        wildcard_regex = re.compile(r'{}\b'.format(token.strip('*')))
    elif token[-1] == '*':
        wildcard_regex = re.compile(r'\b{}'.format(token.strip('*')))
    else:
        # in weird cases, just replace the stars with wildcards
        wildcard_regex = re.compile(r'{}'.format(token.replace('*', '.')))


    return wildcard_vector, wildcard_regex



if __name__ == "__main__":

    s = time.time()
    parse_text_passages_tokens(['aoeue'])
    print(time.time() - s )

