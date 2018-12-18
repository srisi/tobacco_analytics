import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

def store_csr_matrix_to_file(matrix, file_path, compressed=True):

    if compressed:
        np.savez_compressed('{}'.format(file_path), data=matrix.data, indices=matrix.indices, indptr=matrix.indptr,
                 shape=matrix.shape)
    else:
        np.savez('{}'.format(file_path), data=matrix.data, indices=matrix.indices, indptr=matrix.indptr,
                 shape=matrix.shape)

def load_csr_matrix_from_file(file_path):

    y = np.load('{}.npz'.format(file_path))
    matrix = csr_matrix( (y['data'], y['indices'], y['indptr']), shape=y['shape'])

    return matrix


def load_csc_matrix_from_file(file_path):

    y = np.load('{}.npz'.format(file_path))
    matrix = csc_matrix( (y['data'], y['indices'], y['indptr']), shape=y['shape'])

    return matrix

