from unittest import TestCase

import numpy as np
from numpy import testing

from descipher import DES, LeftShifter


class TestDES(TestCase):
    def setUp(self):
        self.des = DES(self.key_matrix)

    key = '0000000100100011010001010110011110001001101010111100110111101111'
    key_matrix = DES.KeyMatrixBuilder(key).build()

    def test_build_key_matrix(self):
        key_matrix = DES.KeyMatrixBuilder(self.key).build()
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

    def test_key_schedule(self):
        np.testing.assert_array_equal(self.des.generate_sub_key(), np.array(
            [[1., 0., 0., 1., 0., 0.],
             [1., 0., 0., 1., 0., 1.],
             [1., 0., 1., 0., 0., 0.],
             [1., 1., 0., 0., 0., 1.],
             [1., 1., 1., 0., 0., 0.],
             [1., 1., 0., 0., 0., 0.],
             [0., 0., 1., 1., 1., 0.],
             [0., 0., 1., 0., 0., 0.]]
        ))

    def test_np_split(self):
        array = np.array([[1, 2, 3],
                          [4, 5, 6]])
        print(np.split(array, 2))
        top, bottom = np.split(array, 2)
        np.testing.assert_array_equal(top, np.array([[1, 2, 3]]))
        np.testing.assert_array_equal(bottom, np.array([[4, 5, 6]]))

    def test_do_round(self):
        self.des.generate_sub_key()
        self.des.do_round()

    def test_convert(self):
        text = '''1 6 7 20 21 29 1 2 28 1 7
1 1 5 23 26 5 1 8 31 1 0
2 8 24 1 4 32 27 3 9
1 9 1 3 30 6 22 1 1 4 25'''
        print(f'''np.array([[{text}]])'''.replace(' ', ',').replace('\n', '],\n['))

    def test_get_index(self):
        array = np.array([[1, 2, 3, 4, 5, 6]])
        print(DES.get_substitution_indexes(array))
        np.testing.assert_array_equal(DES.get_substitution_indexes(array), (7, 14))

    def test_binary(self):
        np.testing.assert_array_equal(DES.to_binary(3), np.array([0, 0, 1, 1]))

    def test_binary_2(self):
        np.testing.assert_array_equal(DES.to_binary(1), np.array([0, 0, 0, 1]))

    def test_get_key(self):
        key = '00010011 00110100 01010111 01111001 10011011 10111100 11011111 11110001'.replace(' ', '')
        print(DES.KeyMatrixBuilder(key).build())

    def test_left_shift(self):
        value = '11110000110011001010101011110101010101100110011110001111'
        value_matrix = DES.KeyMatrixBuilder(value, (1, 56)).build().flatten()
        result = '11100001100110010101010111111010101011001100111100011110'
        result_matrix = DES.KeyMatrixBuilder(result, (1, 56)).build().flatten()
        actual = LeftShifter(1).shift(value_matrix)
        np.testing.assert_array_equal(actual, result_matrix)

    def test_example(self):
        key = np.array([[0, 0, 0, 1, 0, 0, 1, 1],
                        [0, 0, 1, 1, 0, 1, 0, 0],
                        [0, 1, 0, 1, 0, 1, 1, 1],
                        [0, 1, 1, 1, 1, 0, 0, 1],
                        [1, 0, 0, 1, 1, 0, 1, 1],
                        [1, 0, 1, 1, 1, 1, 0, 0],
                        [1, 1, 0, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 0, 0, 0, 1]])
        des = DES(key)
        sub_key = des.generate_sub_key()
        np.testing.assert_array_equal(sub_key, np.array([[0, 0, 0, 1, 1, 0],
                                                         [1, 1, 0, 0, 0, 0],
                                                         [0, 0, 1, 0, 1, 1],
                                                         [1, 0, 1, 1, 1, 1],
                                                         [1, 1, 1, 1, 1, 1],
                                                         [0, 0, 0, 1, 1, 1],
                                                         [0, 0, 0, 0, 0, 1],
                                                         [1, 1, 0, 0, 1, 0]]))

    def test_transform(self):
        key = '''14    17   11    24     1    5
                  3    28   15     6    21   10
                 23    19   12     4    26    8
                 16     7   27    20    13    2
                 41    52   31    37    47   55
                 30    40   51    45    33   48
                 44    49   39    56    34   53
                 46    42   50    36    29   32'''
        print(repr(np.array(key.rsplit()).reshape((8, 6)).astype(int)))
