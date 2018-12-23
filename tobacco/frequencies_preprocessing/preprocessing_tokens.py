import hashlib

import numpy as np
from pathlib import Path
from tobacco.configuration import PATH_TOKENS
from tobacco.frequencies_preprocessing.preprocessing_years_cython import transform_doc_to_year_array
from tobacco.utilities.vector_transformation import csc_to_np_int64, csc_to_np_int32, \
    csc_to_np_uint8
from tobacco.utilities.sparse_matrices import load_csc_matrix_from_file
from tobacco.utilities.vector import Vector


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

    h = hash
    if docs_or_sections == 'sections':
        h += '_sections'
    token_path = Path(PATH_TOKENS, hash[0], hash[1], hash[2], hash[3], h)

#    token_path = PATH_TOKENS + '{}/{}/{}/{}/{}'.format(hash[0], hash[1], hash[2], hash[3], hash)

#    if docs_or_sections == 'sections':
#        token_path += '_sections'

    ngram_vector = Vector()
    ngram_vector.load_from_disk(token_path)
    ngram_vector.convert_to_datatype(return_type)

    return ngram_vector

    from IPython import embed; embed()


#    csc = load_csc_matrix_from_file(token_path)
#    if csc is None:
#        print("Could not find token vector for token {} at {}".format(token, token_path))


    # if return_type == 'csc':
    #     pass
    # elif return_type == 'np':
    #     out = csc_to_np_int64(csc)
    # elif return_type == 'uint8':
    #     out = csc_to_np_uint8(csc)
    #
    # else:
    #     raise ValueError("{} is not a valid return type for get_ngram_vector. 'csc' and 'np' are valid.")

    if return_sum:
        token_sum = csc.data.sum()
        return out, token_sum
    else:
        return out


# def get_tokens_data(token_list, docs_or_sections, active_filters_np):
#     """
#     Load counts, frequencies, and totals for each token.
#
#     12/18/18 Implemented for use with the NgramResult class, i.e. it won't return a df but
#     individual vars.
#
#     :param df: The final results dict
#     :param token_list: list of tokens to process
#     :param docs_or_sections: 'docs' or 'sections'
#     :return: result df with added token data
#
#     >>> from tobacco.frequencies_preprocessing.preprocessing_globals_loader import get_globals
#     >>> from tobacco.frequencies_preprocessing.preprocessing_filters import get_active_filters_np
#     >>> globals = get_globals()
#     >>> active_filters = {'doc_type': [], 'collection': [], 'availability': [], 'term': []}
#     >>> _, _, active_filters_np = get_active_filters_np(active_filters, globals['filters'], 'docs', np.uint8)
#     >>> t = get_tokens_data(['addiction'], 'docs', active_filters_np)
#     >>> t
#
#     """
#
#     tokens_data = []
#
#     # The aggregate is the sum of all vectors, used for the collections and document types.
#     aggregate = None
#
#     for token in token_list:
#         # Load token and totals
#         try:
#             loaded_vector = get_ngram_vector(token, return_type='np_int64',
#                                              docs_or_sections=docs_or_sections)
#
#         except FileNotFoundError:
#             print('Could not load token {}.'.format(token))
#             continue
#
#         # initialize aggregate
#         if aggregate is None:
#             aggregate = loaded_vector.copy()
#         else:
#             aggregate += loaded_vector
#
#         # TODO: vector
#         absolute_counts = transform_doc_to_year_array(data=loaded_vector, filter=active_filters_np,
#                                                       docs_or_sections=docs_or_sections)
#
#         tokens_data.append({
#             'token': token,
#             'counts': absolute_counts,
#             'frequencies': absolute_counts / df['totals_years'],
#             'total': int(np.sum(absolute_counts))
#
#         })
#
#     tokens = sorted(tokens_data, key=lambda k: k['total'], reverse=True)
#
#
#     aggregate *= active_filters_np
#     aggregate_years = transform_doc_to_year_array(data=aggregate, docs_or_sections=docs_or_sections)
#
#     return tokens_data, aggregate, aggregate_years



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

    from tobacco.frequencies_preprocessing.preprocessing_globals_loader import get_globals
    from tobacco.frequencies_preprocessing.preprocessing_filters import get_active_filters_np
    globals = get_globals(load_only_docs=True)
    active_filters = {'doc_type': [], 'collection': [], 'availability': [], 'term': []}
    _, _, active_filters_np = get_active_filters_np(active_filters, globals['filters'], 'docs',
                                                    np.uint8)
    t = get_tokens_data(['addiction'], 'docs', active_filters_np)