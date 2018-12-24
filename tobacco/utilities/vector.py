import numpy as np
from scipy.sparse import csc_matrix
from pathlib import Path
from typing import Union, Tuple
import hashlib

from tobacco.configuration import YEAR_COUNT, DOC_COUNT, SECTION_COUNT, PATH_TOKENS, PATH_TOKENIZED
from IPython import embed

# from tobacco.utilities.vector_transformation import csc_to_np_int64, csc_to_np_int32, \
#     csc_to_np_uint8, load_csc_as_int32, transform_doc_to_year_array_with_filter, \
#     transform_doc_to_year_array_no_filter, transform_doc_to_year_array_no_filter_csc, \
# transform_doc_to_year_array_with_filter_csc, test_uint8
from tobacco.utilities.vector_transformation import *

class Vector:
    """
    class to replace the many vectors used in this project, e.g. filters or term vectors.

    """

    def __init__(self, arr: Union[np.ndarray, csc_matrix]=None):

        self._sum = None
        self._nnz = None

        if arr is None:
            self.datatype = None
            self.vector = None

        elif isinstance(arr, csc_matrix):
            self.datatype = 'csc'
            if not arr.shape[1] == 1:
                raise ValueError(f'Turning a csc_matrix into a Vector requires a 1 dimensional '
                                 f'csc matrix but a {self.shape()[1]} dimensional matrix was '
                                 f'passed.')
            self.vector = arr
            if type(self.vector.data[0]) != np.int32:
                self.vector.data = self.vector.data.astype(np.int32)

        elif isinstance(arr, np.ndarray):
            self.vector = arr
            self._set_datatype_from_np_array()


    def __repr__(self):
        if len(self) == YEAR_COUNT:
            out = '<Year Vector'
        elif len(self) == DOC_COUNT:
            out = '<Document Vector'
        elif len(self) == SECTION_COUNT:
            out = '<Section Vector'
        else:
            out = f'<{len(self)} Vector'

        out += f' of type {self.datatype} with {self.nnz} elements and length {len(self)}.>'

        return out

    def __eq__(self, other):
        """
        Equal if same datatype and same vector

        :param other:
        :return:
        """
        if self.datatype != other.datatype:
            return False
        if self.shape() != other.shape():
            return False
        if self.datatype == 'csc':
            if (np.array_equal(self.vector.data, other.vector.data) and
                    np.array_equal(self.vector.indices, other.vector.indices)):
                return True
            else:
                return False
        else:
            if np.array_equal(self.vector, other.vector):
                return True
            else:
                return False

    def __getitem__(self, key):
        """

        :param key:
        :return:

        >>> csc = csc_matrix(np.array(range(20))).T.tocsc()
        >>> v = Vector(csc)
        >>> v[0:5]
        <5x1 sparse matrix of type '<class 'numpy.int32'>'
    	... with 4 stored elements in Compressed Sparse Column format>

        """

        if self.datatype == 'csc':
            return self.vector[key, 0]
        else:
            return self.vector[key]

    def __add__(self, other):
        """
        Addition works for Vectors and ints

        :param other:
        :return:

        >>> nparr = Vector(np.array(range(5), dtype=np.int32))
        >>> nparr[:]
        array([0, 1, 2, 3, 4], dtype=int32)

        >>> added = nparr + nparr
        >>> added[:]
        array([0, 2, 4, 6, 8], dtype=int32)

        >>> (nparr + 1)[:5]
        array([1, 2, 3, 4, 5], dtype=int32)
        """

        if isinstance(other, int):
            return Vector(self.vector + other)

        elif isinstance(other, Vector):
            if not len(self) == len(other):
                raise ValueError(f'Can only add two Vectors of the same length but not {self} and '
                                 f'{other}.')
            if not self.datatype == other.datatype:
                raise TypeError(f'Can only add two Vectors of the same datatype but not {self} and'
                                f' {other}.')

            return Vector(self.vector + other.vector)

        else:
            raise TypeError(f'Can only add int or Vector to Vector, not {type(other)}.')


    def __truediv__(self, other):
        """

        >>> nparr = Vector(np.array(range(5), dtype=np.int32))
        >>> div_vector = nparr / 10
        >>> div_vector[:]
        array([0. , 0.1, 0.2, 0.3, 0.4])
        >>> type((div_vector)[0])
        <class 'numpy.float64'>

        >>> (nparr / Vector(np.array(range(5, 10), dtype=np.int32)))[:]
        array([0.        , 0.16666667, 0.28571429, 0.375     , 0.44444444])

        :param other:
        :return:
        """

        if isinstance(other, int) or isinstance(other, float):
            return Vector(self.vector / other)

        elif isinstance(other, Vector):
            if not len(self) == len(other):
                raise ValueError(f'Can only divide a Vector by a Vector of the same length '
                                 f'but not {other}.')
            if self.datatype == 'np_uint8':
                raise ValueError(f"Why are you trying to divide a bool vector? ({self})")

            return Vector(self.vector / other.vector)
        else:
            raise NotImplementedError(f'Vector division for {self} is not yet implemented.')




    def copy(self):

        return Vector(self.vector.copy())


    def shape(self) -> Tuple[int, int]:
        """
        Returns the shape of the array, None for non-initialized Vector
        >>> x = Vector().load_token_vector('addiction', return_type='csc')
        >>> x.shape()
        (11303161, 1)

        :return: Tuple[int, int]
        """

        try:
            return self.vector.shape
        except AttributeError:
            print("Warning: Returning shape of non-initialized vector.")
            return (0,0)


    def __len__(self) -> int:
        """
        Returns the number of rows of the vector
        >>> x = Vector().load_token_vector('addiction', return_type='csc')
        >>> len(x)
        11303161

        :return: int
        """
        return self.shape()[0]

    @property
    def sum(self):
        """
        Lazy-loading for Vector sum. Returns 0 for non-initialized Vector
        10ms for numpy int64 array, 0.1ms for csc

        >>> from tobacco.utilities.vector import Vector
        >>> x = Vector().load_token_vector('addiction', return_type='csc')
        >>> x.sum
        945200

        :return: int
        """

        if self._sum is None:

            if not self.datatype:
                print("Warning: Returning sum of non-initialized vector.")
                self._sum = 0
            elif self.datatype == 'csc':
                self._sum =  self.vector.data.sum()
            else:
                self._sum = self.vector.sum()

        return self._sum

    @property
    def nnz(self):
        """
        Lazy-loading for number of non-zero values
        Returns the number of non-zero values
        16ms for numpy int64 array, 0.3ms for csc

        >>> from tobacco.utilities.vector import Vector
        >>> x = Vector().load_token_vector('addiction', return_type='csc')
        >>> x.nnz
        173146

        >>> y = Vector().load_token_vector('addiction', return_type='np_int32')
        >>> y.nnz
        173146

        :return: int
        """
        if self._nnz is None:

            if not self.datatype:
                print("Warning: Returning count_nonzero of non-initialized vector.")
                self._nnz = 0

            elif self.datatype == 'csc':
                self._nnz = len(self.vector.data)
#                self._nnz = self.vector.count_nonzero()
            elif self.datatype == 'np_int32':
                self._nnz = count_nonzero_int32(self.vector)
            elif self.datatype == 'np_uint8':
                self._nnz = count_nonzero_uint8(self.vector)
            elif self.datatype == 'np_float32' or self.datatype == 'np_float64':
                self._nnz = np.count_nonzero(self.vector)
            else:
                raise NotImplementedError("only implemented for csc, int32, and uint8 but not"
                                          f" {self.datatype}.")

        return self._nnz


    def _set_datatype_from_np_array(self):
        """
        Gets the datatype (str) from the numpy array
        (Yes, it might have been better to just use numpy datatypes for that as well. The problem
        is the csc support)

        :return:
        """

        arr_type = type(self.vector[0])
        if arr_type == np.int64:
            self.datatype = 'np_int64'
        elif arr_type == np.int32:
            self.datatype = 'np_int32'
        elif arr_type == np.float64:
            self.datatype = 'np_float64'
        elif arr_type == np.uint8:
            self.datatype = 'np_uint8'
        else:
            raise ValueError(f'Can only initialize Vector with datatypes csc, np.int32/64, '
                             f' float32, and uint8 but not {self.datatype}.')

    def convert_to_datatype(self, output_datatype):

        if output_datatype == self.datatype:
            return self

        if self.datatype == 'csc':

            if output_datatype == 'np_int64':
                self.vector = csc_to_np_int64(self.vector)
                self.datatype = 'np_int64'
                self._sum = None
                self._nnz = None

            elif output_datatype == 'np_int32':
                self.vector = csc_to_np_int32(self.vector)
                self.datatype = 'np_int32'
                self._sum = None
                self._nnz = None

            elif output_datatype == 'np_uint8':
                self.vector = csc_to_np_uint8(self.vector)
                self.datatype = 'np_uint8'
                self._sum = None
                self._nnz = None

            else:
                raise NotImplementedError(f'Vector conversion from {self.datatype} to '
                                          f'{output_datatype} are not available')
        elif output_datatype == 'csc':
            self.vector = convert_int32_to_csc(self)
            #self.vector = csc_matrix(self.vector).T.tocsc()

            self.datatype = 'csc'
            self._sum = None
            self._nnz = None

        else:
            raise NotImplementedError(f'Vector conversion from {self.datatype} to '
                                      f'{output_datatype} are not available')

        return self

    def convert_to_year_array(self, filter_vec=None, copy=True):
        """
        Converts a document or section array into a year array.
        By default, returns a copy of the year array

        >>> v_csc = Vector().load_token_vector('addiction', return_type='csc')
        >>> v_csc_yearly = v_csc.convert_to_year_array()
        >>> print(f'Counts 1950-55: {v_csc_yearly[50:55]}')
        Counts 1950-55: [ 22 131 142 156 433]

        >>> v = Vector().load_token_vector('addiction')
        >>> v_yearly = v.convert_to_year_array()
        >>> print(f'Counts 1950-55: {v_yearly[50:55]}')
        Counts 1950-55: [ 22 131 142 156 433]

        >>> filter_v = Vector().load_token_vector('nicotine', return_type='np_uint8')
        >>> v_yearly_filtered = v.convert_to_year_array(filter_vec=filter_v)
        >>> print(f'Counts of addiction and nicotine intersection 1950-55: {v_yearly_filtered[50:55]}')
        Counts of addiction and nicotine intersection 1950-55: [  7  43 111  99 411]

        >>> v_csc_yearly_filtered = v_csc.convert_to_year_array(filter_vec=filter_v)
        >>> print(f'Counts of addiction and nicotine intersection 1950-55: {v_csc_yearly_filtered[50:55]}')
        Counts of addiction and nicotine intersection 1950-55: [  7  43 111  99 411]

        :param copy: bool
        :return: Vector
        """

        # sum stays stable in some cases but calculation for year vectors is fast -> reset for all.
        self._sum = None
        self._nnz = None

        if len(self) not in {DOC_COUNT, SECTION_COUNT}:
            raise ValueError(f'Can only turn document or section vector into year vector but not '
                             f'vector with len {len(self)}.')
        if self.datatype not in {'np_int32', 'csc'}:
            raise ValueError(f'Can only turn np_int32 vector into year vector, not {repr(self)}.')

        if filter_vec is not None:
            if not isinstance(filter_vec, Vector) or \
                    filter_vec.datatype not in {'np_uint8', 'np_int32', 'csc'}:
                raise ValueError(f'Filter has to be a np_uint8 vector not {filter_vec}.')

            if self.datatype == 'csc':
                if filter_vec.datatype == 'np_uint8':
                    year_array = transform_doc_to_year_csc_with_filter_uint8(self, filter_vec)
                elif filter_vec.datatype == 'np_int32':
                    year_array = transform_doc_to_year_csc_with_filter_int32(self, filter_vec)
                elif filter_vec.datatype == 'csc':
                    year_array = transform_doc_to_year_csc_with_filter_csc(self, filter_vec)
                else:
                    raise NotImplementedError(f'Conversion to year array with {self} and filter '
                                              f'{filter_vec} is not implemented.')
            else:
                if filter_vec.datatype == 'np_uint8':
                    year_array = transform_doc_to_year_int32_with_filter_uint8(self, filter_vec)
                elif filter_vec.datatype == 'np_int32':
                    year_array = transform_doc_to_year_int32_with_filter_int32(self, filter_vec)
                elif filter_vec.datatype == 'csc':
                    year_array = transform_doc_to_year_int32_with_filter_csc(self, filter_vec)
                else:
                    raise NotImplementedError(f'Conversion to year array with {self} and filter '
                                              f'{filter_vec} is not implemented.')
        else:
            if self.datatype == 'csc':
                year_array = transform_doc_to_year_csc_no_filter(self)
            else:
                year_array = transform_doc_to_year_int32_no_filter(self)

        if not copy:
            self.vector = year_array
        else:
            return Vector(year_array)


    def filter_with(self, filter, return_copy=False):
        """
        Filters vector with a boolean vector
        Currently only implemented for np bool vectors

        >>> a = Vector(np.array([1,2,3,4,5], dtype=np.int32))
        >>> filter = Vector(np.array([0,1,0,1,0], dtype=np.uint8))
        >>> a.filter_with(filter)
        >>> a[:5]
        array([0, 2, 0, 4, 0], dtype=int32)

        :param filter:
        :param copy:
        :return:
        """

        if not self.datatype in {'np_uint8', 'np_int32'}:
            raise NotImplementedError(f'Currently, vector filtering is only implemented for np_uint8'
                                      f' and np_int32 but not f{self.datatype}.')
        if not isinstance(filter, Vector) or filter.datatype != 'np_uint8':
            raise ValueError(f'Filter has to be np_uint8 Vector, not {self}.')

        if return_copy:
            out = self.copy()
            out.vector *= filter.vector
            out._nnz = None
            out._sum = None
            return out
        else:
            self._nnz = None
            self._sum = None
            self.vector *= filter.vector


    def load_token_vector(self, token, return_type='np_int32', docs_or_sections='docs'):
        """
        Load a token vector from disk

        :param token:
        :param return_type:
        :param docs_or_sections:
        :return: Vector

        >>> addiction = Vector().load_token_vector('addiction', 'np_int32')
        >>> addiction[0:10]
        array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=int32)

        """

        # to distribute the millions of stored ngram vectors, they were hashed.
        hash = hashlib.sha256(token.encode()).hexdigest()

        filename = hash
        if docs_or_sections == 'sections':
            filename += '_sections'
        token_path = Path(PATH_TOKENS, hash[0], hash[1], hash[2], hash[3], filename)

        self.load_from_disk(token_path, return_type=return_type)
        return self

    def load_filter_vector(self, search_term: Union[str, int], filter_type: str, weight: bool=False,
                           return_type: str='csc', docs_or_sections='docs'):
        '''
        Loads a filter vector

        :param filter_type: 'collection', 'doc_type', or 'availability

        >>> col_vector = Vector().load_filter_vector(5, 'collection', return_type='csc')
        >>> col_vector
        <Document Vector of type csc with 4472485 elements and length 11303161.>

        >>> dt_vector = Vector().load_filter_vector('letter', 'doc_type', return_type='csc')
        >>> dt_vector
        <Document Vector of type csc with 2490726 elements and length 11303161.>

        :return:
        '''

        try:
            # can't store terms with a forward slash -> replace with underscore
            if filter_type == 'doc_type': search_term = search_term.replace('/', '_')
            file_name = '{}_{}_{}_{}'.format(search_term, filter_type, docs_or_sections, weight)
            file_path = Path(PATH_TOKENIZED, 'filters', file_name)
            self.load_from_disk(file_path, return_type=return_type)
            return self
#            filter = Vector().load_from_disk(file_path, return_type=return_type)

        except IOError:
            from tobacco.frequencies_preprocessing.preprocessing_filters import create_filter
            create_filter(search_term, filter_type, weight, return_type, docs_or_sections)
            return self.load_filter_vector(search_term, filter_type, weight, return_type,
                                           docs_or_sections)

    def load_totals_vector(self, search_term, filter_type, docs_or_sections, return_type='csc'):
        """
        >>> v = Vector().load_totals_vector(5, 'collection', 'docs')
        >>> v
        <Document Vector of type csc with 4444260 elements and length 11303161.>

        :param search_term:
        :param filter_type:
        :param docs_or_sections:
        :param return_type:
        :return:
        """


        file_name = '{}_{}'.format(search_term, docs_or_sections)
        file_path = Path(PATH_TOKENIZED, 'totals', file_name)
        return self.load_from_disk(file_path, return_type=return_type)


    def load_from_disk(self, file_path, return_type):
        """
        >>> x = Vector()
        >>> path = '/pcie/tobacco/tokenized/filters/3_collection_docs_False'


        """
        if isinstance(file_path, str):
            file_path = Path(file_path)

        if not return_type in {'csc', 'np_int32', 'np_float32', 'np_uint8'}:
            raise ValueError('Tobacco Analytics should only need csc, np_int32, np_float32, and np_uint8, but'
                             f' not {return_type}.')

        # if the file ends in .npy (google tokens), then it's a numpy array
        if file_path.suffix == '.npy':
            self.vector = np.load(file_path)
            self._set_datatype_from_np_array()


        else:
            self.datatype = 'csc'
            file_path = Path(file_path.parent, f'{file_path.stem}.npz')
            try:
                y = np.load(file_path)
            except FileNotFoundError:
                raise FileNotFoundError(f'Could not load csc vector from {file_path}.')

            data = y['data']
            if not data.dtype == np.int32:
                data = data.astype(np.int32)

            if return_type == 'np_int32':
                self.vector = load_csc_as_int32(y['shape'][0], data, y['indices'])
                self.datatype = 'np_int32'

            else:
                self.vector = csc_matrix((data, y['indices'], y['indptr']), shape=y['shape'])
                self.convert_to_datatype(return_type)

        self._sum = None
        self._nnz = None

        return self

    def save_to_disk(self, file_path, compressed=True):

        if isinstance(file_path, str):
            file_path = Path(file_path)

        if not Path.exists(file_path):
            Path.mkdir(file_path.parent, parents=True)


        if self.datatype == 'csc':
            if compressed:
                np.savez_compressed('{}'.format(file_path), data=self.vector.data,
                                    indices=self.vector.indices, indptr=self.vector.indptr,
                                    shape=self.vector.shape)
            else:
                np.savez('{}'.format(file_path), data=self.vector.data,
                         indices=self.vector.indices, indptr=self.vector.indptr,
                         shape=self.vector.shape)

        elif isinstance(self.vector, np.ndarray):
            np.save(file_path, self.vector)



if __name__ == '__main__':

    pass

    v = Vector().load_token_vector('addiction', return_type='np_int32', docs_or_sections='docs')
    v_csc = Vector().load_token_vector('addiction', return_type='csc', docs_or_sections='docs')
    v_docs = Vector().load_token_vector('addiction', return_type='np_int32', docs_or_sections='docs')


    filter_int32 = Vector().load_filter_vector('letter', filter_type='doc_type', docs_or_sections='sections',
                                             return_type='np_int32')
    filter_uint8 = Vector().load_filter_vector('letter', filter_type='doc_type', docs_or_sections='sections',
                                             return_type='np_uint8')
    filter_csc = Vector().load_filter_vector('letter', filter_type='doc_type', docs_or_sections='sections',
                                             return_type='csc')

    filter_int32 = Vector().load_filter_vector(13, filter_type='collection', docs_or_sections='docs',
                                             return_type='np_int32')
    filter_uint8 = Vector().load_filter_vector(13, filter_type='collection', docs_or_sections='docs',
                                             return_type='np_uint8')
    filter_csc = Vector().load_filter_vector(13, filter_type='collection', docs_or_sections='docs',
                                             return_type='csc')

#    v_csc =

#    x = v_csc.convert_to_year_array(filter_uint8)
#    y = v_csc.convert_to_year_array(filter_int32)



    z = v_csc.convert_to_year_array(filter_csc)
    print(z)
    print(len(np.intersect1d(v_csc.vector.indices, filter_csc.vector.indices)), np.intersect1d(v_csc.vector.indices, filter_csc.vector.indices)[:100])
#    print(v_docs)
#    print(filter_csc)
#    a = v_docs.convert_to_year_array(filter_csc)

    print(x == y, y == z, z == a)

    from IPython import get_ipython
    ipython = get_ipython()
    ipython.magic('%timeit v_csc.convert_to_year_array(filter_uint8)')
    ipython.magic('%timeit v_csc.convert_to_year_array(filter_int32)')
    ipython.magic('%timeit v_csc.convert_to_year_array(filter_csc)')
    ipython.magic('%timeit v_docs.convert_to_year_array(filter_uint8)')
