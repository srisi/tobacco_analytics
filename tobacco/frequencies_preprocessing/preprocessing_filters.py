
import inspect

import numpy as np
from pathlib import Path
from scipy.sparse import csc_matrix

from tobacco.configuration import PATH_TOKENIZED, VALID_COLLECTIONS, DOC_COUNT, SECTION_COUNT
from tobacco.frequencies_preprocessing.preprocessing_doc_types import get_dtype_dict
from tobacco.frequencies_preprocessing.preprocessing_sections import get_doc_id_to_section_id_dict
from tobacco.utilities.databases import Database

from tobacco.utilities.vector import Vector

from typing import Union

def get_active_filters_np(active_filters, FILTERS, docs_or_sections='docs'):

    """ Applies all filters to both the term and a copy of the totals vector and returns them

    6/10/17 Added availability filter.
    The idea is that every time the availability filter is used, the collection and doc type filters get multiplied with it.
    That is to say: the availability filter is not used on its own.

    7/25/17 Added term filter

    >>> FILTERS = {
    ...     'docs':{
    ...         'doc_type': get_doc_type_filters(return_type='csc', docs_or_sections='docs'),
    ...         'collection': get_collection_filters(return_type='csc', docs_or_sections='docs'),
    ...         'availability': get_availability_filters(return_type='csc', docs_or_sections='docs')
    ...     },
    ...     'sections':{
    ...         'doc_type': get_doc_type_filters(return_type='csc', docs_or_sections='sections'),
    ...         'collection': get_collection_filters(return_type='csc', docs_or_sections='sections'),
    ...         'availability': get_availability_filters(return_type='csc', docs_or_sections='sections')
    ...     }
    ... }
    >>> active_filters = {
    ...     'doc_type': {'internal communication'},
    ...     'collection': {2,3},
    ...     'availability': {'no restrictions'},
    ...     'term': {}
    ... }
    >>> doc_type_filter, collection_filter, final_filter = get_active_filters_np(
    ...         active_filters=active_filters, FILTERS=FILTERS, docs_or_sections='docs')


    :param active_filters: dict of lists, e.g. {'doc_type': ["internal communication"], 'collection': [1,2],
                                                'availability': [], 'term': []}
    :param FILTERS: Filters from global ??I think they are csc filters ??
    :param return_type: e.g. np.uint8 or np.int32. By default, the same document type as the input, usually np.uint8
    :param docs_or_sections: 'docs' or 'sections'
    :return: doc_type_filter, collection_filter, final_filter
    """

    if not 'term' in active_filters:
        active_filters['term'] = {}

    # all filters used here are unweighted
    # 8/31/18: At some point, I had the idea that a document with 10 document types would give weight 1/10 to each.
    weighted = False
    filter_len = DOC_COUNT
    if docs_or_sections == 'sections':
        filter_len = SECTION_COUNT

    # process availability filters
    if len(active_filters['availability']) == 0:
        availability_filter = None
    else:
        availability_filter = None
        for filter_name in active_filters['availability']:
            if availability_filter is None:
                availability_filter = FILTERS[docs_or_sections]['availability'][(filter_name, weighted)].copy()
            else:
                availability_filter += FILTERS[docs_or_sections]['availability'][(filter_name, weighted)]
        availability_filter.convert_to_datatype('np_uint8')

    # process term filters
    if len(active_filters['term']) == 0:
        term_filter = None
    else:
        term_filter = None
        for filter_name in active_filters['term']:
            if term_filter is None:
                term_filter = Vector().load_token_vector(filter_name, return_type='np_uint8',
                                                         docs_or_sections=docs_or_sections)
            else:
                term_filter += Vector().load_token_vector(filter_name, return_type='np_uint8',
                                                          docs_or_sections=docs_or_sections)

    # process doc_type filters
    if len(active_filters['doc_type']) == 0:
        doc_type_filter = np.ones(filter_len, dtype='bool')
    else:
        doc_type_filter = None
        for filter_name in active_filters['doc_type']:
            if doc_type_filter is None:

                doc_type_filter = FILTERS[docs_or_sections]['doc_type'][(filter_name, weighted)].copy()
            else:
                doc_type_filter += FILTERS[docs_or_sections]['doc_type'][(filter_name, weighted)]

        # if docs_or_sections == 'sections':
        doc_type_filter.convert_to_datatype('np_uint8')

    # process collection filters
    if len(active_filters['collection']) == 0:
        collection_filter = np.ones(filter_len, dtype=np.uint8)
    else:
        collection_filter = None
        for filter_name in active_filters['collection']:
            if collection_filter is None:
                collection_filter = FILTERS[docs_or_sections]['collection'][(filter_name, weighted)].copy()
            else:
                collection_filter += FILTERS[docs_or_sections]['collection'][(filter_name, weighted)]
        collection_filter.convert_to_datatype('np_uint8')

    # Apply term filter to doc type and collection filters
    if term_filter is not None:
        doc_type_filter.filter_with(term_filter)
        collection_filter.filter_with(term_filter)

    # Apply availability filter to doc type and collection filters
    if availability_filter is not None:
        doc_type_filter.filter_with(availability_filter)
        doc_type_filter.filter_with(availability_filter)

    # Create final filter
    if len(active_filters['doc_type']) == 0:
        final_filter = collection_filter
    elif len(active_filters['collection']) == 0:
        final_filter = doc_type_filter
    else:
        final_filter = collection_filter.filter_with(doc_type_filter, return_copy=True)

    return doc_type_filter, collection_filter, final_filter


def create_filter(search_term: Union[str, int], filter_type: str, weight: bool,
                  return_type: str, docs_or_sections: str):

    '''
    Creates a filter vector for collectinos, doc types, availability

    12/20/18: Separated from get_filter (moved to Vector class)

    :param search_term:
    :param filter_type:
    :param weight:
    :param return_type:
    :param docs_or_sections:
    :return:
    '''

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
            for section_id in range(first_id, last_id + 1):
                filter[section_id] = True


    filter_vec = Vector(csc_matrix(filter))
    if filter_type == 'doc_type':
        search_term = search_term.replace('/', '_')
    file_name = '{}_{}_{}_{}'.format(search_term, filter_type, docs_or_sections, weight)
    file_path = Path(PATH_TOKENIZED, 'filters', file_name)
    filter_vec.save_to_disk(file_path)



#    filter_path = PATH_TOKENIZED + 'filters/{}_{}_{}_{}.npz'.format(search_term, filter_type, docs_or_sections, weight)
#    store_csr_matrix_to_file(filter, filter_path)
    print("Created filter for {} with {} elements.".format(search_term, filter_vec.vector.getnnz()))


def get_filter(search_term, filter_type, weight=False, return_type='csc', docs_or_sections='docs'):
    '''
    Creates a binary filter (True if document has the specificed doc_type or collection. Falso otherwise
    :param search_term:
    :param filter_type: 'collection' or 'doc_type'
    :param weight:
    :return: Vector

    >>> filter = get_filter('letter', 'doc_type', weight=False, return_type='csc')
    >>> filter
    <Document Vector of type csc with 2490726 elements.>

    '''


    try:
        # can't store terms with a forward slash -> replace with underscore
        if filter_type == 'doc_type': search_term = search_term.replace('/', '_')
        file_name = '{}_{}_{}_{}'.format(search_term, filter_type, docs_or_sections, weight)
        file_path = Path(PATH_TOKENIZED, 'filters', file_name)
        filter = Vector().load_from_disk(file_path, return_type=return_type)

#        filter =  load_csc_matrix_from_file(PATH_TOKENIZED + 'filters/{}_{}_{}_{}'.format(search_term, filter_type, docs_or_sections, weight))


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

        return get_filter(filter_type, weight, return_type, docs_or_sections)

#    if return_type == 'np':
#        filter = np.array(filter.todense()).flatten()

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
                collection_filters[(collection_id, False)] = get_filter(collection_id,
                        filter_type='collection', return_type='np_uint8',
                        docs_or_sections=docs_or_sections)
            else:
                collection_filters[(collection_id, False)] = get_filter(collection_id,
                        filter_type='collection', return_type='csc',
                        docs_or_sections=docs_or_sections)

        else:
            collection_filters[(collection_id, False)] = get_filter(collection_id,
                                filter_type='collection', return_type=return_type,
                                docs_or_sections=docs_or_sections)

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
                doc_type_filters[(doc_type, weight)] = get_filter(doc_type, filter_type='doc_type',
                          weight=weight, return_type=return_type, docs_or_sections=docs_or_sections)
            else:
                doc_type_filters[(doc_type, weight)] = get_filter(doc_type, filter_type='doc_type',
                          weight=weight, return_type='csc', docs_or_sections=docs_or_sections)


    for group in doc_types['groups']:
        for weight in [False]:

            file_name = '{}_{}_{}_{}'.format(group, 'doc_type', docs_or_sections, weight)
            file_path = Path(PATH_TOKENIZED, 'filters', file_name)
            try:
                group_filter = Vector().load_from_disk(file_path, return_type=return_type)

            except FileNotFoundError:
                print("creating group filter for: ", group)
                group_filter = None
                for doc_type in doc_types['groups'][group]:
                    if group_filter is None:
                        group_filter = doc_type_filters[(doc_type, weight)]
                    else:
                        group_filter += doc_type_filters[(doc_type, weight)]

                group_filter.save_to_disk(file_path)

            doc_type_filters[(group, weight)] = group_filter
#            if return_type == 'np':
#                doc_type_filters[(group, weight)] = csc_bool_to_np_cython(group_filter)

    return doc_type_filters


def get_filters(return_type='csc'):
    '''

    :param return_type: 'csc' or 'np'
    :return:
    '''

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

    from IPython import embed
    FILTERS = {
        'docs':{
            'doc_type': get_doc_type_filters(return_type='csc', docs_or_sections='docs'),
            'collection': get_collection_filters(return_type='csc', docs_or_sections='docs'),
            'availability': get_availability_filters(return_type='csc', docs_or_sections='docs')
        },
#        'sections':{
#            'doc_type': get_doc_type_filters(return_type='csc', docs_or_sections='sections'),
#            'collection': get_collection_filters(return_type='csc', docs_or_sections='sections'),
#            'availability': get_availability_filters(return_type='csc', docs_or_sections='sections')
#        }
    }
    active_filters = {
        'doc_type': {'internal communication'},
        'collection': {2,3},
        'availability': {'no restrictions'},
        'term': {}
    }
    doc_type_filter, collection_filter, final_filter = get_active_filters_np(
            active_filters=active_filters, FILTERS=FILTERS, docs_or_sections='docs')