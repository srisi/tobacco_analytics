"""
Contains functions to quickly transform 1d csc matrices (i.e. csc vectors) into numpy vectors

"""

import numpy as np
cimport numpy as np

DTYPE_FLOAT64 = np.float64
ctypedef np.float64_t DTYPE_FLOAT64_t
DTYPE_INT = np.int
ctypedef np.int_t DTYPE_INT_t


from cython cimport boundscheck, wraparound

@wraparound(False)
@boundscheck(False)
def csc_to_np_int64(csc_vector):
    """ Given a sparse csc vector, creates a 1d numpy array with the same data in dtype int32

    :param csc_vector:
    :return:
    """

    cdef np.ndarray[int, ndim=1] indices = csc_vector.indices
    cdef np.ndarray[long, ndim=1] data = csc_vector.data

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

@wraparound(False)
@boundscheck(False)
def csc_bool_to_np_cython(csc_vector):
    '''
    Given a sparse csc vector, creates a 1d numpy array with the same data in dtype int32

    10/10/2018 Unclear if both csc_bool_to_np_cython and csc_to_np_uint8 are needed. Prob not.

    :param csc_vector:
    :return:
    '''

    cdef np.ndarray[int, ndim=1] indices = csc_vector.indices
#    cdef np.ndarray[np.uint8_t, ndim=1] data = csc_vector.data.view(dtype=np.uint8)

    cdef int index, count, i
    cdef np.ndarray[np.uint8_t, ndim=1 ] out = np.zeros(csc_vector.shape[0], dtype=np.uint8)

    for i in range(len(indices)):
        # 7/31/17: sholudn't this be = 1 instead of data[i]? i.e. eliminate that one lookup.
        out[indices[i]] = 1

#    print("csc_bool_to_np_cython for vector shaped {} took {}.".format(out.shape, time.time()-s))

    return out


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
    cdef np.ndarray[long, ndim=1] data = csc_vector.data

    cdef int index, count, i, cur_index
    cdef np.ndarray[np.uint8_t, ndim=1] out = np.zeros(csc_vector.shape[0], dtype=np.uint8)
    cdef int out_len = len(out)

    for i in range(len(indices)):
        out[indices[i]] = 1

    return out
