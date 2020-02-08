import unittest

from tobacco.utilities.vector import Vector

import numpy as np
from scipy.sparse import csc_matrix

vec_csc = Vector().load_token_vector('addiction', docs_or_sections='docs', return_type='csc')
vec_int32 = Vector().load_token_vector('addiction', docs_or_sections='docs', return_type='np_int32')
vec_uint8 = Vector().load_token_vector('addiction', docs_or_sections='docs', return_type='np_uint8')

filter_csc = Vector().load_filter_vector('letter', filter_type='doc_type', docs_or_sections='docs',
                                         return_type='csc')
filter_int32 = Vector().load_filter_vector('letter', filter_type='doc_type', docs_or_sections='docs',
                                         return_type='np_int32')
filter_uint8 = Vector().load_filter_vector('letter', filter_type='doc_type', docs_or_sections='docs',
                                         return_type='np_uint8')

class TestVectorInit(unittest.TestCase):

    def test_vector_initialization(self):
        """
        Make sure that the vectors are initialized with the correct datatypes
        """
        self.assertTrue(isinstance(vec_csc.vector.data[0], np.int32))
        self.assertEqual(vec_csc.datatype, 'csc')
        self.assertTrue(isinstance(filter_csc.vector.data[0], np.int32))
        self.assertEqual(filter_csc.datatype, 'csc')

        self.assertTrue(isinstance(vec_int32.vector[0], np.int32))
        self.assertEqual(vec_int32.datatype, 'np_int32')
        self.assertTrue(isinstance(filter_int32.vector[0], np.int32))
        self.assertEqual(filter_int32.datatype, 'np_int32')

        self.assertTrue(isinstance(vec_uint8.vector[0], np.uint8))
        self.assertEqual(vec_uint8.datatype, 'np_uint8')
        self.assertTrue(isinstance(filter_uint8.vector[0], np.uint8))
        self.assertEqual(filter_uint8.datatype, 'np_uint8')


    def test_conversion_to_year_csc_no_filter(self):
        self.assertEqual(vec_csc.convert_to_year_array(), vec_int32.convert_to_year_array())

    def test_conversion_to_year_int32_with_filter(self):
        self.assertEqual(vec_int32.convert_to_year_array(filter_vec=filter_int32),
                         vec_int32.convert_to_year_array(filter_vec=filter_uint8))
        self.assertEqual(vec_int32.convert_to_year_array(filter_vec=filter_int32),
                         vec_int32.convert_to_year_array(filter_vec=filter_csc))

    # def test_conversion_to_year_uint8_with_filter(self):
    #     self.assertEqual(vec_uint8.convert_to_year_array(filter_vec=filter_int32),
    #                      vec_uint8.convert_to_year_array(filter_vec=filter_uint8))
    #     self.assertEqual(vec_uint8.convert_to_year_array(filter_vec=filter_int32),
    #                      vec_uint8.convert_to_year_array(filter_vec=filter_csc))

    def test_conversion_to_year_csc_with_filter(self):
        self.assertEqual(vec_csc.convert_to_year_array(filter_vec=filter_int32),
                         vec_csc.convert_to_year_array(filter_vec=filter_uint8))
        self.assertEqual(vec_csc.convert_to_year_array(filter_vec=filter_int32),
                         vec_csc.convert_to_year_array(filter_vec=filter_csc))





if __name__ == '__main__':
    unittest.main()