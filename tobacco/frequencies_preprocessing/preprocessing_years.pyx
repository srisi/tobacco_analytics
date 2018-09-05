import pickle

from scipy.sparse import lil_matrix, csc_matrix
from tobacco.configuration import PATH_TOKENIZED, DOC_COUNT, YEAR_COUNT, SECTION_COUNT, YEAR_START, YEAR_END
from tobacco.frequencies_preprocessing.preprocessing_sections import get_doc_id_to_section_id_dict
from tobacco.utilities.databases import Database
from tobacco.utilities.sparse_matrices import store_csr_matrix_to_file, load_csc_matrix_from_file


def get_year_doc_transformation_matrix(docs_or_sections='docs'):
    """ Returns a csc matrix (M) used to turn a 12 million term count (t) vector into a len 116 year vector (y)

    M * t = y

    M -> One row for each year, One col for each document.
    M(x,y) = 1 if doc y is from year x. 0 otherwise.

    8/31/18: This was a nice way of doing it but I don't think the matrix is needed anymore because the simpler
    multiply_and_transform_doc_to_year fulfills the same function less elegantly but faster.

    :param docs_or_sections: 'docs' or 'sections'
    :return:
    """

    try:
        year_doc_matrix = load_csc_matrix_from_file(PATH_TOKENIZED + 'year_doc_matrix_{}'.format(docs_or_sections))
        if not year_doc_matrix.dtype == np.uint8:
            year_doc_matrix = csc_matrix(year_doc_matrix, dtype=np.uint8)

        # print("year doc", docs_or_sections)
        # print(type(year_doc_matrix), year_doc_matrix.dtype)
        # print(year_doc_matrix.sum(), year_doc_matrix.nnz)

    except IOError:


        print("Year Doc Transformation matrix not available for {}. Creating now...".format(docs_or_sections))

        doc_id_to_section_id_dict = get_doc_id_to_section_id_dict()

        db = Database("TOB_FULL")
        con, cur = db.connect()

        n = DOC_COUNT
        if docs_or_sections == 'sections':
            n = SECTION_COUNT

        year_doc_matrix = lil_matrix((YEAR_COUNT, n))

        for year in range(YEAR_START, YEAR_END+1):
            print(year)
            cur.execute("SELECT MIN(id) FROM docs WHERE year = {}".format(year))
            min_id = cur.fetchall()[0]['MIN(id)']
            cur.execute("SELECT MAX(id) FROM docs WHERE year = {}".format(year))
            max_id = cur.fetchall()[0]['MAX(id)']

            row = year - YEAR_START

            if docs_or_sections == 'sections':
                min_id = doc_id_to_section_id_dict[min_id][0]
                max_id = doc_id_to_section_id_dict[max_id][1]


            year_doc_matrix[row, min_id: max_id + 1] = 1

        year_doc_matrix = year_doc_matrix.tocsc()
        year_doc_matrix = csc_matrix(year_doc_matrix, dtype=np.uint8)
        print("year_doc_matrix has {} elements. Shape: {}.".format(year_doc_matrix.getnnz(), year_doc_matrix.shape))
        store_csr_matrix_to_file(year_doc_matrix, PATH_TOKENIZED + 'year_doc_matrix_{}.npz'.format(docs_or_sections))

        return year_doc_matrix

def get_year_doc_id_list(docs_or_sections):

    """ Returns a list, wherein every value marks the first doc_id belonging to that year.
    e.g. year_doc_id_list[1910] -> first id belonging to year 1910
    year_doc_id_list[2015] -> highest doc_id + 1

    :param docs_or_sections: 'docs' or 'sections'
    :return:
    """

    doc_id_to_section_id_dict = get_doc_id_to_section_id_dict()

    try:
        year_doc_id_list = pickle.load(open(PATH_TOKENIZED + 'year_{}_id_list.pickle'.format(docs_or_sections), 'rb'))

    except IOError:

        print("Creating new year_{}_id_list".format(docs_or_sections))

        db = Database("TOB_FULL")
        con, cur = db.connect()

        year_doc_id_list = []

        for year in range(1901, 2017):
            cur.execute("SELECT MIN(id), MAX(id) FROM docs WHERE year = {}".format(year))
            row = cur.fetchall()[0]
            min_doc_id = row['MIN(id)']
            max_doc_id = row['MAX(id)']

            if docs_or_sections == 'docs':
                year_doc_id_list.append((min_doc_id, max_doc_id))
                print(year, min_doc_id)
            elif docs_or_sections == 'sections':
                min_section_id = doc_id_to_section_id_dict[min_doc_id][0]
                max_section_id = doc_id_to_section_id_dict[max_doc_id][1]
                year_doc_id_list.append((min_section_id, max_section_id))
                print(year, min_section_id, max_section_id)


        pickle.dump(year_doc_id_list, open(PATH_TOKENIZED + 'year_{}_id_list.pickle'.format(docs_or_sections), 'wb'))

    return year_doc_id_list


import numpy as np
cimport numpy as np

DTYPE_FLOAT64 = np.float64
ctypedef np.float64_t DTYPE_FLOAT64_t
DTYPE_INT = np.int
ctypedef np.int_t DTYPE_INT_t

YEAR_PARTS_ID_LIST = {
    'docs': get_year_doc_id_list('docs'),
    'sections': get_year_doc_id_list('sections')
}

from cython cimport boundscheck, wraparound


def transform_doc_to_year_array(data, docs_or_sections, filter=None):
    """ Cython implemented transformation from doc or section vector to year vector

    i.e. if we have an 11 million length document vector that stores the counts of addiction, this function
    returns the 116 length year vector that covers the years 1901 to 2016.

    Idea: store the first and last id of every year. Then iterate over all years.
    For every document or section, check if the filter includes it. If it does, add document/section count to the
    total for that year.

    There are two different functions, one with filter and one without. This function selects the appropriate cython
    function to use (cython has static types so filter cannot be either None or an np array).

    :param data: np array of ints, len either number of documents or sections
    :param filter: None or np.uint8 bool array
    :param docs_or_sections: 'docs' or 'sections'
    """

    if filter:
        return transform_doc_to_year_array_filter(data, filter, docs_or_sections)
    else:
        return transform_doc_to_year_array_no_filter(data, docs_or_sections)

@wraparound(False)
@boundscheck(False)
def transform_doc_to_year_array_filter(np.ndarray[int, ndim=1] data, np.ndarray[dtype=np.uint8_t, ndim=1] filter,
                                       str docs_or_sections):
    """

    :param data:
    :param filter:
    :param docs_or_sections:
    :return:
    """

    cdef int year_sum
    cdef unsigned long doc_id, year_id, start_id, end_id

    cdef np.ndarray[int, ndim=1] years = np.zeros(YEAR_COUNT, dtype=np.int32)


    for year_id in range(YEAR_COUNT):
        start_id, end_id = YEAR_PARTS_ID_LIST[docs_or_sections][year_id]
        year_sum = 0
        for doc_id in range(start_id, end_id + 1):
            if filter[doc_id]:
                year_sum += data[doc_id]
        years[year_id] = year_sum

    return years

@wraparound(False)
@boundscheck(False)
def transform_doc_to_year_array_no_filter(np.ndarray[int, ndim=1] data, str docs_or_sections):
    """

    :param data:
    :param docs_or_sections:
    :return:
    """

    cdef int year_sum
    cdef unsigned long doc_id, year_id, start_id, end_id

    cdef np.ndarray[int, ndim=1] years = np.zeros(YEAR_COUNT, dtype=np.int32)


    for year_id in range(YEAR_COUNT):
        start_id, end_id = YEAR_PARTS_ID_LIST[docs_or_sections][year_id]
        year_sum = 0
        for doc_id in range(start_id, end_id + 1):
            year_sum += data[doc_id]
        years[year_id] = year_sum

    return years


if __name__ == "__main__":
    #    get_year_doc_transformation_matrix()
    pass