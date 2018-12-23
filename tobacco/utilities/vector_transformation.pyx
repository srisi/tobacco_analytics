"""
Contains functions to quickly transform 1d csc matrices (i.e. csc vectors) into numpy vectors

"""

from tobacco.configuration import YEAR_COUNT, DOC_COUNT, SECTION_COUNT
from tobacco.frequencies_preprocessing.preprocessing_years import get_year_doc_id_list

from scipy.sparse import csc_matrix

from cpython cimport array
import array

import numpy as np
cimport numpy as np

DTYPE_FLOAT64 = np.float64
ctypedef np.float64_t DTYPE_FLOAT64_t
DTYPE_INT = np.int
ctypedef np.int_t DTYPE_INT_t

from cython cimport boundscheck, wraparound

YEAR_PARTS_ID_LIST = {
    'docs': get_year_doc_id_list('docs'),
    'sections': get_year_doc_id_list('sections')
}

@wraparound(False)
@boundscheck(False)
def csc_to_np_int64(csc_vector):
    """ Given a sparse csc vector, creates a 1d numpy array with the same data in dtype int32

    :param csc_vector:
    :return:
    """

    cdef np.ndarray[int, ndim=1] indices = csc_vector.indices
    cdef np.ndarray[int, ndim=1] data = csc_vector.data

    cdef int index, count, i
    cdef np.ndarray[int, ndim=1 ] out = np.zeros(csc_vector.shape[0], dtype=np.int32)

    for i in range(len(indices)):
        out[indices[i]] = data[i]

    return out

@wraparound(False)
@boundscheck(False)
def csc_to_np_int32(csc_vector):
    '''
    Given a sparse csc vector, creates a 1d numpy array with the same data in dtype int32

    :param csc_vector:
    :return:
    '''

    cdef np.ndarray[int, ndim=1] indices = csc_vector.indices
    cdef np.ndarray[int, ndim=1] data = csc_vector.data

    cdef int index, count, i
    cdef np.ndarray[int, ndim=1 ] out = np.zeros(csc_vector.shape[0], dtype=np.int32)

    for i in range(len(indices)):
        out[indices[i]] = data[i]

    return out

# @wraparound(False)
# @boundscheck(False)
# def np_int64_to_np_int32(np.ndarray[long, ndim=1] np_arr):
#
#     cdef np.ndarray[int, ndim=1] new_arr = np.zeros(len(np_arr), dtype=np.int32)
#     cdef int i
#     for i in range(len(np_arr)):
#         new_arr[i] = np_arr[i]
#     return new_arr

# @wraparound(False)
# @boundscheck(False)
# def csc_bool_to_np_uint8(csc_vector):
#     '''
#     Given a sparse csc vector, creates a 1d numpy array with the same data in dtype int32
#
#     10/10/2018 Unclear if both csc_bool_to_np_cython and csc_to_np_uint8 are needed. Prob not.
#
#     :param csc_vector:
#     :return:
#     '''
#
#     cdef np.ndarray[int, ndim=1] indices = csc_vector.indices
# #    cdef np.ndarray[np.uint8_t, ndim=1] data = csc_vector.data.view(dtype=np.uint8)
#
#     cdef int index, count, i
#     cdef np.ndarray[np.uint8_t, ndim=1 ] out = np.zeros(csc_vector.shape[0], dtype=np.uint8)
#
#     for i in range(len(indices)):
#         out[indices[i]] = 1
#     return out


@wraparound(False)
@boundscheck(False)
def csc_to_np_uint8(csc_vector):
    '''
    Given a sparse csc sections vector, this function returns a 1d numpy array
    such that for every section with a positive value, that section as well as the preceeding and
    suceeding sections are set to True or 1
    The idea is to limit down which docmuents and sections text passage search needs to consider.

    [0, 0, 1, 0, 0, 0, 0, 1, 1] -> [0, 1, 1, 1, 0, 0, 1, 1, 1]
    '''

    cdef np.ndarray[int, ndim=1] indices = csc_vector.indices
    cdef np.ndarray[int, ndim=1] data = csc_vector.data

    cdef int index, count, i, cur_index
    cdef np.ndarray[np.uint8_t, ndim=1] out = np.zeros(csc_vector.shape[0], dtype=np.uint8)
    cdef int out_len = len(out)

    for i in range(len(indices)):
        out[indices[i]] = 1

    return out

@wraparound(False)
@boundscheck(False)
def transform_doc_to_year_int32_with_filter_uint8(vector, filter_vec):
        # np.ndarray[int, ndim=1] data,
        #                                np.ndarray[dtype=np.uint8_t, ndim=1] filter):
    """

    :param data:
    :param filter:
    :param docs_or_sections:
    :return:
    """

    cdef int year_sum
    cdef unsigned long doc_id, year_id, start_id, end_id

    cdef np.ndarray[int, ndim=1] years = np.zeros(YEAR_COUNT, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] data = vector.vector
    cdef np.ndarray[dtype=np.uint8_t, ndim=1] filter = filter_vec.vector

    cdef str docs_or_sections = 'docs'
    if len(vector) == SECTION_COUNT:
        docs_or_sections = 'sections'


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
def transform_doc_to_year_int32_with_filter_int32(vector, filter_vec):
    """

    :param data:
    :param filter:
    :param docs_or_sections:
    :return:
    """

    cdef int year_sum
    cdef unsigned long doc_id, year_id, start_id, end_id

    cdef np.ndarray[int, ndim=1] years = np.zeros(YEAR_COUNT, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] data = vector.vector
    cdef np.ndarray[int, ndim=1] filter = filter_vec.vector

    cdef str docs_or_sections = 'docs'
    if len(vector) == SECTION_COUNT:
        docs_or_sections = 'sections'


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
def transform_doc_to_year_int32_with_filter_csc(vector, filter_vec):
    """

    :param data:
    :param filter:
    :param docs_or_sections:
    :return:
    """

    cdef int year_sum
    cdef int doc_id, year_id, start_id, end_id, cur_filter_indices_idx, cur_filter_doc_idx

    cdef np.ndarray[int, ndim=1] years = np.zeros(YEAR_COUNT, dtype=np.int32)

    cdef np.ndarray[int, ndim=1] data = vector.vector
    cdef np.ndarray[int, ndim=1] filter_data = filter_vec.vector.data
    cdef np.ndarray[int, ndim=1] filter_indices = filter_vec.vector.indices
    cdef int len_filter_indices = len(filter_indices)

    cur_filter_indices_idx = 0
    cur_filter_doc_idx = filter_indices[0]

#    print()
#    print(filter_vec)
#    print(filter_vec.vector)

    cdef str docs_or_sections = 'docs'
    if len(vector) == SECTION_COUNT:
        docs_or_sections = 'sections'


    for year_id in range(YEAR_COUNT):
        start_id, end_id = YEAR_PARTS_ID_LIST[docs_or_sections][year_id]

#        print(year_id+1901, start_id, end_id, cur_filter_doc_idx)

        # go into loop if start_id <= cur_doc_idx <= end_id and not all indices have been processed.
        if (cur_filter_doc_idx >= start_id and cur_filter_doc_idx <= end_id and cur_filter_indices_idx < len_filter_indices):
            year_sum = 0
            while True:
                if cur_filter_doc_idx > end_id:
                    break
                else:
                    year_sum += data[cur_filter_doc_idx]
#                    print(year_sum, data[cur_filter_doc_idx])
                    cur_filter_indices_idx += 1
                    if len_filter_indices == cur_filter_indices_idx:
                        break
                    else:
                        cur_filter_doc_idx = filter_indices[cur_filter_indices_idx]
            years[year_id] = year_sum
#            print(year_id+1901, year_sum)
        else:
            years[year_id] = 0

    return years



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
def transform_doc_to_year_int32_no_filter(vector):
    """

    :param data:
    :param docs_or_sections:
    :return:
    """

    cdef int year_sum
    cdef int doc_id, year_id, start_id, end_id

    cdef np.ndarray[int, ndim=1] data = vector.vector
    cdef np.ndarray[int, ndim=1] years = np.zeros(YEAR_COUNT, dtype=np.int32)

    cdef str docs_or_sections = 'docs'
    if len(vector) == SECTION_COUNT:
        docs_or_sections = 'sections'


    for year_id in range(YEAR_COUNT):
        start_id, end_id = YEAR_PARTS_ID_LIST[docs_or_sections][year_id]
        year_sum = 0
        for doc_id in range(start_id, end_id + 1):
            year_sum += data[doc_id]
        years[year_id] = year_sum

    return years

@wraparound(False)
@boundscheck(False)
def transform_doc_to_year_csc_no_filter(vector):

    # cur_indices_idx -> current id of the indices array
    # cur_doc_idx -> current doc_id (i.e. where the indices array is pointing to
    cdef int doc_id, year_id, start_id, end_id, year_sum, cur_indices_idx, cur_doc_idx


    cdef np.ndarray[int, ndim=1] indices = vector.vector.indices
    cdef np.ndarray[int, ndim=1] data = vector.vector.data
    cdef int len_indices = len(indices)

    cdef np.ndarray[int, ndim=1] years = np.zeros(YEAR_COUNT, dtype=np.int32)

    cdef str docs_or_sections = 'docs'
    if len(vector) == SECTION_COUNT:
        docs_or_sections = 'sections'

    cur_indices_idx = 0
    cur_doc_idx = indices[0]

    for year_id in range(YEAR_COUNT):
        start_id, end_id = YEAR_PARTS_ID_LIST[docs_or_sections][year_id]

        # go into loop if start_id <= cur_doc_idx <= end_id and not all indices have been processed.
        if cur_doc_idx >= start_id and cur_doc_idx <= end_id and cur_indices_idx < len_indices:

            year_sum = 0
            while True:
                if cur_doc_idx > end_id:
                    break
                else:
                    year_sum += data[cur_indices_idx]
                    cur_indices_idx += 1
                    if len_indices == cur_indices_idx:
                        break
                    else:
                        cur_doc_idx = indices[cur_indices_idx]
            years[year_id] = year_sum
        else:
            years[year_id] = 0

    return years


@wraparound(False)
@boundscheck(False)
def transform_doc_to_year_csc_with_filter_uint8(vector, filter_vec):

    # cur_indices_idx -> current id of the indices array
    # cur_doc_idx -> current doc_id (i.e. where the indices array is pointing to
    cdef int doc_id, year_id, start_id, end_id, year_sum, cur_indices_idx, cur_doc_idx


    cdef np.ndarray[int, ndim=1] indices = vector.vector.indices
    cdef np.ndarray[int, ndim=1] data = vector.vector.data
    cdef np.ndarray[dtype=np.uint8_t, ndim=1] filter = filter_vec.vector
    cdef int len_indices = len(indices)

    cdef np.ndarray[int, ndim=1] years = np.zeros(YEAR_COUNT, dtype=np.int32)

    cdef str docs_or_sections = 'docs'
    if len(vector) == SECTION_COUNT:
        docs_or_sections = 'sections'

    cur_indices_idx = 0
    cur_doc_idx = indices[0]

    for year_id in range(YEAR_COUNT):
        start_id, end_id = YEAR_PARTS_ID_LIST[docs_or_sections][year_id]

        # go into loop if start_id <= cur_doc_idx <= end_id and not all indices have been processed.
        if cur_doc_idx >= start_id and cur_doc_idx <= end_id and cur_indices_idx < len_indices:

            year_sum = 0
            while True:
                if cur_doc_idx > end_id:
                    break
                else:
                    if filter[cur_doc_idx]:
                        year_sum += data[cur_indices_idx]
                    cur_indices_idx += 1
                    if len_indices == cur_indices_idx:
                        break
                    else:
                        cur_doc_idx = indices[cur_indices_idx]
            years[year_id] = year_sum
        else:
            years[year_id] = 0

    return years


@wraparound(False)
@boundscheck(False)
def transform_doc_to_year_csc_with_filter_int32(vector, filter_vec):

    # cur_indices_idx -> current id of the indices array
    # cur_doc_idx -> current doc_id (i.e. where the indices array is pointing to
    cdef int doc_id, year_id, start_id, end_id, year_sum, cur_indices_idx, cur_doc_idx


    cdef np.ndarray[int, ndim=1] indices = vector.vector.indices
    cdef np.ndarray[int, ndim=1] data = vector.vector.data
    cdef np.ndarray[int, ndim=1] filter = filter_vec.vector
    cdef int len_indices = len(indices)

    cdef np.ndarray[int, ndim=1] years = np.zeros(YEAR_COUNT, dtype=np.int32)

    cdef str docs_or_sections = 'docs'
    if len(vector) == SECTION_COUNT:
        docs_or_sections = 'sections'

    cur_indices_idx = 0
    cur_doc_idx = indices[0]

    for year_id in range(YEAR_COUNT):
        start_id, end_id = YEAR_PARTS_ID_LIST[docs_or_sections][year_id]

        # go into loop if start_id <= cur_doc_idx <= end_id and not all indices have been processed.
        if cur_doc_idx >= start_id and cur_doc_idx <= end_id and cur_indices_idx < len_indices:

            year_sum = 0
            while True:
                if cur_doc_idx > end_id:
                    break
                else:
                    if filter[cur_doc_idx]:
                        year_sum += data[cur_indices_idx]
                    cur_indices_idx += 1
                    if len_indices == cur_indices_idx:
                        break
                    else:
                        cur_doc_idx = indices[cur_indices_idx]
            years[year_id] = year_sum
        else:
            years[year_id] = 0

    return years


@wraparound(False)
@boundscheck(False)
def transform_doc_to_year_csc_with_filter_csc(vector, filter_vec):

    # cur_indices_idx -> current id of the indices array
    # cur_doc_idx -> current doc_id (i.e. where the indices array is pointing to
    cdef int doc_id, year_id, start_id, end_id, year_sum, cur_indices_idx, cur_doc_idx
    cdef int cur_filter_indices_idx, cur_filter_doc_idx

    cdef np.ndarray[int, ndim=1] indices = vector.vector.indices
    cdef np.ndarray[int, ndim=1] data = vector.vector.data
    cdef np.ndarray[int, ndim=1] filter_indices = filter_vec.vector.indices
    cdef int len_indices = len(indices)
    cdef int len_filter_indices = len(filter_indices)

    cdef np.ndarray[int, ndim=1] years = np.zeros(YEAR_COUNT, dtype=np.int32)

    cdef str docs_or_sections = 'docs'
    if len(vector) == SECTION_COUNT:
        docs_or_sections = 'sections'

    cur_indices_idx = 0
    cur_doc_idx = indices[0]
    cur_filter_indices_idx = 0
    cur_filter_doc_idx = filter_indices[0]

    #find first intersection
    while True:
        if cur_doc_idx == cur_filter_doc_idx:
            break
        elif cur_doc_idx < cur_filter_doc_idx:
            cur_indices_idx += 1
            if len_indices == cur_indices_idx:
                break
            else:
                cur_doc_idx = indices[cur_indices_idx]
        else:
            cur_filter_indices_idx += 1
            if len_filter_indices == cur_filter_indices_idx:
                break
            else:
                cur_filter_doc_idx = filter_indices[cur_filter_indices_idx]



    for year_id in range(YEAR_COUNT):
        start_id, end_id = YEAR_PARTS_ID_LIST[docs_or_sections][year_id]

        # go into loop if start_id <= cur_doc_idx <= end_id and not all indices have been processed.
        if (cur_doc_idx >= start_id and cur_doc_idx <= end_id
                and cur_filter_doc_idx >= start_id and cur_filter_doc_idx <= end_id
                and cur_indices_idx < len_indices and cur_filter_indices_idx < len_filter_indices):


            year_sum = 0
            while True:
                if cur_doc_idx > end_id or cur_filter_doc_idx > end_id:
                    break
                else:
                    year_sum += data[cur_indices_idx]
                    cur_indices_idx += 1
                    cur_filter_indices_idx += 1
                    if len_indices >= cur_indices_idx or len_filter_indices >= cur_filter_indices_idx:
                        break

                    # find next intersecting element
                    else:
                        cur_doc_idx = indices[cur_indices_idx]
                        cur_filter_doc_idx = filter_indices[cur_filter_indices_idx]

                        while True:
                            if cur_doc_idx == cur_filter_doc_idx:
                                break
                            elif cur_doc_idx < cur_filter_doc_idx:
                                cur_indices_idx += 1
                                if len_indices == cur_indices_idx:
                                    break
                                else:
                                    cur_doc_idx = indices[cur_indices_idx]
                            else:
                                cur_filter_indices_idx += 1
                                if len_filter_indices == cur_filter_indices_idx:
                                    break
                                else:
                                    cur_filter_doc_idx = filter_indices[cur_filter_indices_idx]

            years[year_id] = year_sum
        else:
            years[year_id] = 0

    return years



@wraparound(False)
@boundscheck(False)
def load_csc_as_int32(int length, np.ndarray[int, ndim=1] data ,np.ndarray[int, ndim=1] indices):
    """
    Loads the arrays of a csc matrix into a np int32 array.
    Very similar to csc_to_int32

    :param length:
    :param data:
    :param indices:
    :return:
    """

    cdef int index, count, i
    cdef np.ndarray[int, ndim=1 ] out = np.zeros(length, dtype=np.int32)

    for i in range(len(indices)):
        out[indices[i]] = data[i]

    return out

@wraparound(False)
@boundscheck(False)
def count_nonzero_int32(int[:] data):
#def count_nonzero_int32(np.ndarray[int, ndim=1] data):
    """
    Count nonzero elements because numpy takes 20ms for a 10 million array
    (this still takes 10ms... I don't know why.
    :return:
    """

    cdef int i, count_nonzero=0

    for i in range(len(data)):
        if data[i] != 0:
            count_nonzero += 1
    return count_nonzero

def count_nonzero_uint8(np.ndarray[np.uint8_t, ndim=1] data):
    """
    Count nonzero elements because numpy takes 20ms for a 10 million array
    :return:
    """

    cdef int i, count_nonzero=0

    for i in range(len(data)):
        if data[i] != 0:
            count_nonzero += 1
    return count_nonzero


def convert_int32_to_csc(vector):
    """
    Turn a numpy array into a csc matrix
    :param nparr:
    :return:
    """

    cdef np.ndarray[int, ndim=1] nparr = vector.vector
    cdef np.ndarray[int, ndim=1] data = np.empty(vector.nnz, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] indices = np.empty(vector.nnz, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] indptr = np.array([0, vector.nnz], dtype=np.int32)
    cdef tuple csc_shape = (len(nparr), 1)

    cdef int cur_indices = 0, val, i


    for i in range(len(nparr)):
        val = nparr[i]
        if val != 0:
            data[cur_indices] = val
            indices[cur_indices] = i
            cur_indices += 1


    return csc_matrix((data, indices, indptr), shape=csc_shape)
