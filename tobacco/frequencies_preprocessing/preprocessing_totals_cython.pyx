import time

import numpy as np
cimport numpy as np

from scipy.sparse import csc_matrix

DTYPE_FLOAT64 = np.float64
ctypedef np.float64_t DTYPE_FLOAT64_t
DTYPE_INT = np.int
ctypedef np.int_t DTYPE_INT_t

from scipy.sparse import csc_matrix, csr_matrix
from tobacco.utilities.databases import Database
from tobacco.frequencies_preprocessing.preprocessing_tokens import get_ngram_vector
from tobacco.configuration import DOC_COUNT, SECTION_COUNT

# rewrite by summing over all column vectors


def create_totals_vector(int ngram, str docs_or_sections):
    """ Creates doc or section vector with the sum of all token vectors.

    Creating a totals vector would have taken too long w/o cython I think
    8/31/18 Not sure why we need this really...

    :param ngram:
    :return:
    """

    print("Totals vector not available to load. Creating now.")

    start_time = time.time()

    if docs_or_sections == 'docs':
        no_rows = DOC_COUNT
    elif docs_or_sections == 'sections':
        no_rows = SECTION_COUNT

    cdef np.ndarray[long, ndim=1] totals_array = np.zeros(no_rows, dtype=np.int64)
    cdef np.ndarray[long, ndim=1] token_vector_data
    cdef np.ndarray[int, ndim=1]  token_vector_indices

    cdef int i, doc_count = 0

    cdef nnz_processed = 0
    time_section = time.time()

    db = Database("TOB_FULL")
    con, cur = db.connect()

    cur.execute('SELECT token from tokens where ngram=1;')
    while True:
        row = cur.fetchone()
        if not row: break

        doc_count += 1

        if doc_count % 1000 == 0:
            print(doc_count, nnz_processed, time.time() - time_section)
            time_section = time.time()
            nnz_processed = 0


        token = row['token']
        token_vector = get_ngram_vector(token, docs_or_sections=docs_or_sections)
        token_vector_data = token_vector.data
        token_vector_indices = token_vector.indices

        nnz_processed += len(token_vector_data)

        for i in range(len(token_vector_data)):
            totals_array[token_vector_indices[i]] += token_vector_data[i]

    totals_vector = csr_matrix(totals_array).T
    print(totals_vector.shape, totals_vector.getnnz(), type(totals_vector))

    print("Creating totals vector for {}-grams took: {}".format(ngram, time.time()-start_time))

    return totals_vector
