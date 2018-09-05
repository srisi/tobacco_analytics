from cpython cimport array
import array


def tokenize_section_cython(str section, dict vocabulary, array.array indices, array.array indptr):

    cdef str token
    cdef int token_id

    for token in section.split():

        try:
            token_id = vocabulary[token]
            indices.append(token_id)

        except KeyError:
            pass

    indptr.append(len(indices))

    return indices, indptr


if __name__ == '__main__':
    pass
