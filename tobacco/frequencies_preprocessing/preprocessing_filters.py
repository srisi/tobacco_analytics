
from tobacco.utilities.databases import Database
from tobacco.utilities.sparse_matrices import store_csr_matrix_to_file, load_csc_matrix_from_file, csc_bool_to_np_cython
from tobacco.frequencies_preprocessing.preprocessing_doc_types import get_dtype_dict
from tobacco.frequencies_preprocessing.preprocessing_sections import get_doc_id_to_section_id_dict

from tobacco.configuration import PATH_TOKENIZED, VALID_COLLECTIONS, DOC_COUNT, SECTION_COUNT


import numpy as np
import inspect

from scipy.sparse import csc_matrix


def get_filter(search_term, filter_type, weight=False, return_type='csc', docs_or_sections='docs'):
    '''
    Creates a binary filter (True if document has the specificed doc_type or collection. Falso otherwise
    :param search_term:
    :param filter_type: 'collection' or 'doc_type'
    :param weight:
    :return:
    '''




    try:
        # can't store terms with a forward slash -> replace with underscore
        if filter_type == 'doc_type': search_term = search_term.replace('/', '_')
        filter =  load_csc_matrix_from_file(PATH_TOKENIZED + 'filters/{}_{}_{}_{}'.format(search_term, filter_type, docs_or_sections, weight))


    except IOError:


        db = Database("TOB_FULL")
        con, cur = db.connect()

        if docs_or_sections == 'docs':
            filter_len = DOC_COUNT
        elif docs_or_sections == 'sections':
            filter_len = SECTION_COUNT
        else:
            raise ValueError("param docs_or_sections has to be either 'docs' or sections' but not ''{}".format(docs_or_sections))

        if weight:
            filter = np.zeros((filter_len, 1), dtype=np.float)
        else:
            filter = np.zeros((filter_len, 1), dtype=np.bool)


        if filter_type == 'collection':
            cur.execute("SELECT id, no_tokens from docs where collection_id = '{}' ORDER BY id ASC".format(search_term))
        elif filter_type == 'doc_type':
            if weight:
                cur.execute('''SELECT doc_types.doc_id as id, docs.no_tokens as no_tokens, doc_types.weight as weight
                                  FROM doc_types, docs WHERE doc_type = "{}" and doc_types.doc_id = docs.id ORDER BY id ASC'''.format(search_term))
            else:
                cur.execute('SELECT doc_types.doc_id as id, docs.no_tokens as no_tokens FROM doc_types, docs '
                            '     WHERE doc_type = "{}" and doc_types.doc_id = docs.id ORDER BY id ASC'.format(search_term))
        elif filter_type == 'availability':
            # dict maps from search term to the requisite where clause (documents can be both formerly privileged and formerly confidential)
            term_to_mysql_where_clauses_dict = {
                'no restrictions': 'WHERE availability = "public;no restrictions"',
                'formerly confidential': 'WHERE availability = "public;formerly confidential" OR availability = "public;formerly confidential; formerly privileged" OR availability = "public;formerly privileged; formerly confidential"',
                'formerly privileged': 'WHERE availability = "public;formerly privileged" OR availability = "public;formerly confidential; formerly privileged" OR availability = "public;formerly privileged; formerly confidential"'
            }
            cur.execute('SELECT id, no_tokens from docs {} ORDER BY id ASC'.format(term_to_mysql_where_clauses_dict[search_term]))
        else:
            raise KeyError("{} is not a valid filter_type. Valid filter types are 'collection', 'doc_type', and 'availability'".format(filter_type))

        if docs_or_sections == 'sections':
            doc_id_to_section_dict = get_doc_id_to_section_id_dict()

        rows = cur.fetchall()
        for row in rows:
            if docs_or_sections == 'docs':
                if weight:
                    filter[row['id']] = row['weight']
                else:
                    filter[row['id'], 0] = True

            elif docs_or_sections == 'sections':
                first_id, last_id = doc_id_to_section_dict[row['id']]
                for section_id in range(first_id, last_id+1):
                    filter[section_id] = True

        filter = csc_matrix(filter)
        if filter_type == 'doc_type': search_term = search_term.replace('/', '_')

        filter_path = PATH_TOKENIZED + 'filters/{}_{}_{}_{}.npz'.format(search_term, filter_type, docs_or_sections, weight)
        store_csr_matrix_to_file(filter, filter_path)
        print("Created filter for {} with {} elements.".format(search_term, filter.getnnz()))

    if return_type == 'np':
        filter = np.array(filter.todense()).flatten()

    return filter



def get_collection_filters(return_type='csc', docs_or_sections='docs'):
    '''

    :param return_type: 'csc', 'np', or 'mixed'. Mixed returns only the most important collections as np
    :param docs_or_sections:
    :return:
    '''

    collection_filters = {}

    for collection_id in VALID_COLLECTIONS:

        if return_type == 'mixed':
            if collection_id in [3, 5, 6, 7, 8, 9, 10, 11, 15, 18, 19]:
                collection_filters[(collection_id, False)] = get_filter(collection_id, filter_type='collection',
                                                                    return_type='np', docs_or_sections=docs_or_sections)
            else:
                collection_filters[(collection_id, False)] = get_filter(collection_id, filter_type='collection',
                                                                    return_type='csc', docs_or_sections=docs_or_sections)

        else:
            collection_filters[(collection_id, False)] = get_filter(collection_id, filter_type='collection',
                                                                    return_type=return_type, docs_or_sections=docs_or_sections)

    collection_filters[('msa_bat', False)] = collection_filters[(5, False)] + collection_filters[(6, False)] + \
                                    collection_filters[(7, False)] + collection_filters[(8, False)] + \
                                    collection_filters[(9, False)] + collection_filters[(10, False)] + \
                                    collection_filters[(11, False)] + collection_filters[(15, False)]


    return collection_filters

def get_availability_filters(return_type='csc', docs_or_sections = 'docs'):

    availability_filters = {}
    for availability in ['no restrictions', 'formerly confidential', 'formerly privileged']:
        availability_filters[(availability, False)] = get_filter(availability, filter_type='availability',
                                                                 return_type=return_type, docs_or_sections=docs_or_sections)

    return availability_filters


def get_doc_type_filters(return_type='csc', docs_or_sections = 'docs'):


    doc_types = get_dtype_dict()


    doc_type_filters = {}
    for doc_type in doc_types['valid']:
        for weight in [False]:
            if doc_type in [
                'letter', 'report', 'memo', 'email', 'note', 'publication', 'report, scientific', 'advertisement',
                'promotional material', 'budget', 'specification', 'budget_review', 'telex', 'news_article', 'agenda',
                'report, market research', 'speech', 'presentation', 'minutes'
            ]:
                doc_type_filters[(doc_type, weight)] = get_filter(doc_type, filter_type='doc_type', weight=weight,
                                                                  return_type=return_type, docs_or_sections=docs_or_sections)
            else:
                        doc_type_filters[(doc_type, weight)] = get_filter(doc_type, filter_type='doc_type', weight=weight,
                                                              return_type='csc', docs_or_sections=docs_or_sections)


    for group in doc_types['groups']:
        for weight in [False]:

            try:
                group_filter = load_csc_matrix_from_file(PATH_TOKENIZED + 'filters/{}_{}_{}_{}'.format(group, 'doc_type',
                                                                                                       docs_or_sections, weight))

            except FileNotFoundError:
                print("creating group filter for: ", group)
                group_filter = None
                for doc_type in doc_types['groups'][group]:
                    if group_filter is None:
                        group_filter = doc_type_filters[(doc_type, weight)]
                    else:
                        group_filter += doc_type_filters[(doc_type, weight)]

                store_csr_matrix_to_file(group_filter, PATH_TOKENIZED + 'filters/{}_{}_{}_{}.npz'.format(group,'doc_type',
                                                                                                      docs_or_sections, weight))

            doc_type_filters[(group, weight)] = group_filter
            if return_type == 'np':
#                doc_type_filters[(group, weight)] = np.array(group_filter.todense()).flatten()
                doc_type_filters[(group, weight)] = csc_bool_to_np_cython(group_filter)

    return doc_type_filters


def get_filters(return_type='csc'):
    '''

    :param return_type: 'csc' or 'np'
    :return:
    '''

    print("getting filters", inspect.getouterframes( inspect.currentframe() ))

    filters = {
        'docs':{
            'doc_type': get_doc_type_filters(return_type=return_type, docs_or_sections='docs'),
            'collection': get_collection_filters(return_type=return_type, docs_or_sections='docs'),
            'availability': get_availability_filters(return_type=return_type, docs_or_sections='docs')
        },
        'sections':{
            'doc_type': get_doc_type_filters(return_type='csc', docs_or_sections='sections'),
            'collection': get_collection_filters(return_type='csc', docs_or_sections='sections'),
            'availability': get_availability_filters(return_type='csc', docs_or_sections='sections')
        }
    }

    return filters

if __name__ == "__main__":

    pass

#    get_filter('no restrictions', 'availability', False, 'np')
#    get_filters(return_type='np')
#    get_collection_filters(docs_or_sections='sections')
#    get_collection_filters(docs_or_sections='docs')
    test()