import hashlib

import numpy as np
from tobacco.configuration import PATH_TOKENS
from tobacco.frequencies_preprocessing.preprocessing_years import transform_doc_to_year_array
from tobacco.utilities.sparse_matrices import csc_to_np_cython, csc_to_np_uint8
from tobacco.utilities.sparse_matrices import load_csc_matrix_from_file


def get_ngram_vector(token, return_type='csc', return_sum=False, docs_or_sections='docs'):
    """ Loads the ngram vector of the token

    E.g. v = get_ngram_vector('nicotine', return_type='csc', docs_or_sections='docs')

    :param token: search token, string
    :param return_type: 'csc', 'np', 'uint8'
    :param return_sum: Whether or not to return the sum of the vector.
    :param docs_or_sections: 'docs' or 'sections'
    :return:
    """

    # to distribute the millions of stored ngram vectors, they were hashed.
    hash = hashlib.sha256(token.encode()).hexdigest()
    token_path = PATH_TOKENS + '{}/{}/{}/{}/{}'.format(hash[0], hash[1], hash[2], hash[3], hash)

    if docs_or_sections == 'sections':
        token_path += '_sections'

    csc = load_csc_matrix_from_file(token_path)
    if csc is None:
        print("Could not find token vector for token {} at {}".format(token, token_path))

    if return_type == 'csc':
        out = csc
    elif return_type == 'np':
        out = csc_to_np_cython(csc)
    elif return_type == 'uint8':
        out = csc_to_np_uint8(csc)

    else:
        raise ValueError("{} is not a valid return type for get_ngram_vector. 'csc' and 'np' are valid.")

    if return_sum:
        token_sum = csc.data.sum()
        return out, token_sum
    else:
        return out


def get_tokens(df, token_list, docs_or_sections):
    """ Load counts, frequencies, and totals for each token. Returns df with token data

    :param df: The final results dict
    :param token_list: list of tokens to process
    :param docs_or_sections: 'docs' or 'sections'
    :return: result df with added token data
    """

    tokens = []

    # The aggregate is the sum of all vectors, used for the collections and document types.
    aggregate = None

    for token in token_list:
        # Load token and totals
        try:
            loaded_vector = get_ngram_vector(token, return_type='np', docs_or_sections=docs_or_sections)

        except FileNotFoundError:
            print('Could not load token {}.'.format(token))
            continue

        # initialize aggregate
        if aggregate is None:
            aggregate = loaded_vector.copy()
        else:
            aggregate += loaded_vector

        absolute_counts = transform_doc_to_year_array(data=loaded_vector, filter=df['active_filters_np'],
                                                      docs_or_sections=docs_or_sections)

        tokens.append({
            'token': token,
            'counts': absolute_counts,
            'frequencies': absolute_counts / df['totals_years'],
            'total': int(np.sum(absolute_counts))

        })

    tokens = sorted(tokens, key=lambda k: k['total'], reverse=True)

    aggregate *= df['active_filters_np']
    aggregate_years = transform_doc_to_year_array(data=aggregate, docs_or_sections=docs_or_sections)

    df['tokens'] = tokens
    df['aggregate'] = aggregate
    df['aggregate_years'] = aggregate_years

    return df



if __name__ == "__main__":

    import time

#    print(get_ngram_vector('addiction').shape)
#    print(get_ngram_vector('addiction', use_sections=True, return_type='np').shape)
    s = time.time()
#    v1 = get_ngram_vector('nicotine', return_type='text_passages_term_filter', docs_or_sections='sections')
    v1 = get_ngram_vector('nicotine', return_type='csc', docs_or_sections='docs')
    print(time.time() - s)

    print(v1[120:150])

    v2 = get_ngram_vector('nicotine', return_type='uint8', docs_or_sections='sections')
    print(v2[120:150])
    pass
