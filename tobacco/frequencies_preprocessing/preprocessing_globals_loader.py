'''
Both text passages and frequencies use many of the same globals (filter etc.), so this scripts loads them all at once.

'''

import time

from tobacco.frequencies_preprocessing.preprocessing_collections import get_col_name_and_idx_dict
from tobacco.frequencies_preprocessing.preprocessing_doc_types import get_doc_types_to_idx_dict
from tobacco.frequencies_preprocessing.preprocessing_filters import get_doc_type_filters, \
    get_collection_filters, get_availability_filters
from tobacco.frequencies_preprocessing.preprocessing_sections import \
    get_section_to_doc_and_offset_arr
from tobacco.frequencies_preprocessing.preprocessing_totals import get_totals_vector
from tobacco.frequencies_preprocessing.preprocessing_years import get_year_doc_id_list
from tobacco.utilities.ocr import get_vocabulary_totals, load_vocabulary_trie
from tobacco.utilities.vector_transformation import csc_to_np_int32



# csc_to_np for sections takes about 40ms -> load more as csc and use transforms.

def get_globals(globals_type='frequencies', load_only_docs=False):
    """
    Returns all the globals necessary to process a frequencies or text_passages global
    use load_only_docs to load only the docs (but not section) filters and totals in frequency mode


    Mode: frequencies, load: docs and sections. 14s
    Mode: frequencies, load: docs                2s

    :param globals_type:
    :return:
    """

    s = time.time()
    if globals_type == 'frequencies':
        globals_dict = {
            'filters': {
                'docs': {
                    'doc_type': get_doc_type_filters(return_type='csc', docs_or_sections='docs'),       #  66 MB
                    'collection': get_collection_filters(return_type='csc', docs_or_sections='docs'),   # 121 MB
                    'availability': get_availability_filters(return_type='csc', docs_or_sections='docs')#  40 MB
                },
            },

            'totals': {
                'totals': {
                    'docs': {
                        'np': get_totals_vector(docs_or_sections='docs', return_type='np_int32'),        # 43 MB
                    },
                },
            },

            'vocabulary_totals': get_vocabulary_totals(1),
            'vocabulary_trie': load_vocabulary_trie(1),  # 1 MB
            'vocabulary_set': load_vocabulary_trie(1, return_type='set'),  # 15 MB

            'year_parts_id_list': {
                'docs': get_year_doc_id_list('docs'),  # 45 MB
            }
        }

        if not load_only_docs:
            globals_dict['filters']['sections'] = {
                'doc_type': get_doc_type_filters(return_type='csc', docs_or_sections='sections'),        # 1009 MB
                'collection': get_collection_filters(return_type='csc', docs_or_sections='sections'),    #  828 MB
                'availability': get_availability_filters(return_type='csc', docs_or_sections='sections') #  427 MB
            }
            globals_dict['totals']['totals']['sections'] = {}
            globals_dict['totals']['totals']['sections']['np'] = get_totals_vector(docs_or_sections='sections', return_type='csc')  # 341 MB
            globals_dict['year_parts_id_list']['sections'] = get_year_doc_id_list('sections')           # 1 MB ??this seems wrong ??





#     if globals_type == 'frequencies':
#         globals_dict = {
#             'filters':{
#                 'docs':{
#                     'doc_type': get_doc_type_filters(return_type='csc', docs_or_sections='docs'),            # 66 MB
#                     'collection': get_collection_filters(return_type='csc', docs_or_sections='docs'),        # 121 MB
#                     'availability': get_availability_filters(return_type='csc', docs_or_sections='docs')     #  40 MB
#                    },
#                 'sections':{
#                     'doc_type': get_doc_type_filters(return_type='csc', docs_or_sections='sections'),       # 1009 MB
#                     'collection': get_collection_filters(return_type='csc', docs_or_sections='sections'),   # 828 MB
#                     'availability': get_availability_filters(return_type='csc', docs_or_sections='sections')# 427 MB
#                 }
#             },
#
#             'totals':{
#                 'totals':{
#                     'docs':{
#                         'np': csc_to_np_int32(get_totals_vector(docs_or_sections='docs')),                 # 43 MB
#                     },
#                     'sections':{
#                         'np': csc_to_np_int32(get_totals_vector(docs_or_sections='sections')),             # 341 MB
#                     },
#                 },
#             },
#
#             # 8/31/18 I don't think these are used -> commented out for the time being.
# #            'year_doc_matrix':{
# #                'docs': get_year_doc_transformation_matrix(docs_or_sections='docs'),                        # 170 MB
# #                'sections': get_year_doc_transformation_matrix(docs_or_sections='sections')                 # 1360 MB
# #            },
#
#             'vocabulary_totals': get_vocabulary_totals(1),
#             'vocabulary_trie': load_vocabulary_trie(1),                                                     # 1 MB
#             'vocabulary_set': load_vocabulary_trie(1, return_type='set'),                                   # 15 MB
#
#             'year_parts_id_list':{
#                 'docs': get_year_doc_id_list('docs'),                                                       # 45 MB
#                 'sections': get_year_doc_id_list('sections')                                                # 1 MB ?? why so much less than docs??
#             }
#         }
    elif globals_type == 'passages':
        globals_dict = {
            'filters':{
                'sections':{
                    'doc_type': get_doc_type_filters(return_type='csc', docs_or_sections='sections'),       # 958 MB
                    'collection': get_collection_filters(return_type='csc', docs_or_sections='sections'),   # 828 MB
                    'availability': get_availability_filters(return_type='csc', docs_or_sections='sections')# 427 MB
                }
            },

            'doc_types_and_idx_dict': get_doc_types_to_idx_dict(),
            'collections_and_idx_dict': get_col_name_and_idx_dict(),
            'section_to_doc_and_offset_arr': get_section_to_doc_and_offset_arr(),                           # 1024 MB

            'vocabulary_totals': get_vocabulary_totals(1),
            'vocabulary_trie': load_vocabulary_trie(1),                                                     # 1 MB
            'vocabulary_set': load_vocabulary_trie(1, return_type='set'),                                   # 15 MB

            'year_parts_id_list':{
                'docs': get_year_doc_id_list('docs'),                                                       # 45 MB
                'sections': get_year_doc_id_list('sections')                                                # 1 MB ?? why so much less than docs??
            }
        }
    else:
        raise ValueError("only 'frequencies' and 'passages' are valid values for globals_type but not {}".format(globals_type))

#    print("Loading globals in {} mode took: {}".format(globals_type, time.time() - s))

    return globals_dict

if __name__ == '__main__':
    pass

#    from IPython import embed
#    embed()
    get_globals(globals_type='frequencies', load_only_docs=True)