from tobacco.configuration import YEAR_COUNT
from tobacco.frequencies_preprocessing.preprocessing_years import get_year_doc_id_list

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


def transform_doc_to_year_array(data: np.ndarray[int], docs_or_sections: str, filter=None):
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

    if filter is None:
        return transform_doc_to_year_array_no_filter(data, docs_or_sections)
    else:
        return transform_doc_to_year_array_filter(data, filter, docs_or_sections)

@wraparound(False)
@boundscheck(False)
def transform_doc_to_year_array_filter(np.ndarray[int, ndim=1] data,
                                       np.ndarray[dtype=np.uint8_t, ndim=1] filter,
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