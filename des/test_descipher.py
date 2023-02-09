from unittest import TestCase

import numpy as np
from numpy import testing

from descipher import DES


class TestDES(TestCase):
    key = '0000000100100011010001010110011110001001101010111100110111101111'

    def test_build_key_matrix(self):
        key_matrix = DES.KeyMatrixBuilder(self.key,
                                          64).build()
        expected_key_matrix = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                                        [0, 0, 1, 0, 0, 0, 1, 1],
                                        [0, 1, 0, 0, 0, 1, 0, 1],
                                        [0, 1, 1, 0, 0, 1, 1, 1],
                                        [1, 0, 0, 0, 1, 0, 0, 1],
                                        [1, 0, 1, 0, 1, 0, 1, 1],
                                        [1, 1, 0, 0, 1, 1, 0, 1],
                                        [1, 1, 1, 0, 1, 1, 1, 1]])
        np.testing.assert_array_equal(expected_key_matrix, key_matrix)

    def test_matrix_indexing(self):
        key_matrix = np.array([[6, 5],
                               [3, 2]])
        value_matrix = np.array([[1, 2, 3],
                                 [4, 5, 6]])
        matrix_indexer = DES.MatrixIndexer(key_matrix)
        np.testing.assert_array_equal(matrix_indexer.convert(value_matrix), np.array([[6, 5], [3, 2]]))

    def test_get_lr_key(self):
        des = DES(self.key)
        print(des.get_l_r_key()[0])
        print(des.get_l_r_key()[1])
        print(des.generate_sub_key()[0])

        print(des.generate_sub_key()[1])
