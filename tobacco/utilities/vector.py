import numpy as np
from scipy.sparse import csc_matrix

class Vector:
    """
    class to replace the many vectors used in this project, e.g. filters or term vectors.

    """

    def __init__(self):

        self.datatype = None

        self.vector = None


    def save_to_disk(self, file_path, compressed=True):

        if self.datatype == 'csc':
            if compressed:
                np.savez_compressed('{}'.format(file_path), data=self.vector.data,
                                    indices=self.vector.indices, indptr=self.vector.indptr,
                                    shape=self.vector.shape)
            else:
                np.savez('{}'.format(file_path), data=self.vector.data,
                         indices=self.vector.indices, indptr=self.vector.indptr,
                         shape=self.vector.shape)


    def load_from_disk(self, file_path):

        if self.datatype == 'csc':
            y = np.load('{}.npz'.format(file_path))
            self.vector = csc_matrix((y['data'], y['indices'], y['indptr']), shape=y['shape'])
