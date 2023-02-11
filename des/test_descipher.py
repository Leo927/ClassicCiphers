from unittest import TestCase

import numpy as np
from numpy import testing

from descipher import DES, LeftShifter, PC2, PC1, Expansion, XOR, IP, SSub, PermutationP
from utils.matrixindexer import MatrixIndexer


class TestDES(TestCase):
    def setUp(self):
        self.des = DES(self.key_matrix, self.value)

    key = '0000000100100011010001010110011110001001101010111100110111101111'
    value = '0000000100100011010001010110011110001001101010111100110111101111'
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
        matrix_indexer = MatrixIndexer(key_matrix)
        np.testing.assert_array_equal(matrix_indexer.convert(value_matrix), np.array([[6, 5], [3, 2]]))

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

    def test_get_key(self):
        key = '00010011 00110100 01010111 01111001 10011011 10111100 11011111 11110001'
        print(DES.KeyMatrixBuilder(key).build())

    def test_PC1(self):
        value = '00010011 00110100 01010111 01111001 10011011 10111100 11011111 11110001'
        value_matrix = DES.KeyMatrixBuilder(value, (1, 64)).build().flatten()
        result = ' 1111000 0110011 0010101 0101111 0101010 1011001 1001111 0001111'
        result_matrix = DES.KeyMatrixBuilder(result, (1, 56)).build().flatten()
        actual = PC1().apply(value_matrix)
        np.testing.assert_array_equal(actual, result_matrix)

    def test_left_shift(self):
        value = '11110000110011001010101011110101010101100110011110001111'
        value_matrix = DES.KeyMatrixBuilder(value, (1, 56)).build().flatten()
        result = '11100001100110010101010111111010101011001100111100011110'
        result_matrix = DES.KeyMatrixBuilder(result, (1, 56)).build().flatten()
        actual = LeftShifter(1).shift(value_matrix)
        np.testing.assert_array_equal(actual, result_matrix)

    def test_PC2(self):
        value = '11100001100110010101010111111010101011001100111100011110'
        value_matrix = DES.KeyMatrixBuilder(value, (1, 56)).build().flatten()
        result = '000110110000001011101111111111000111000001110010'.replace(' ', '')
        result_matrix = DES.KeyMatrixBuilder(result, (1, 48)).build().flatten()
        actual = PC2.apply(value_matrix)
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
        value = '0000 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1111'
        value_matrix = DES.KeyMatrixBuilder(value).build().flatten()
        des = DES(key, value_matrix)
        sub_key = des.generate_sub_key()
        np.testing.assert_array_equal(sub_key, np.array([[0, 0, 0, 1, 1, 0],
                                                         [1, 1, 0, 0, 0, 0],
                                                         [0, 0, 1, 0, 1, 1],
                                                         [1, 0, 1, 1, 1, 1],
                                                         [1, 1, 1, 1, 1, 1],
                                                         [0, 0, 0, 1, 1, 1],
                                                         [0, 0, 0, 0, 0, 1],
                                                         [1, 1, 0, 0, 1, 0]]))
        ciphertext = des.do_round()
        expected_ciphertext = '1111 0000 1010 1010 1111 0000 1010 10101110 1111 0100 1010 0110 0101 0100 0100'
        expected_ciphertext_matrix = DES.KeyMatrixBuilder(expected_ciphertext, (1, 64)).build().flatten()
        np.testing.assert_array_equal(ciphertext, expected_ciphertext_matrix)

    def test_expansion(self):
        value = '1111 0000 1010 1010 1111 0000 1010 1010'
        value_matrix = DES.KeyMatrixBuilder(value, (1, 32)).build().flatten()
        result = ' 011110 100001 010101 010101 011110 100001 010101 010101'.replace(' ', '')
        result_matrix = DES.KeyMatrixBuilder(result, (1, 48)).build().flatten()
        actual = Expansion.apply(value_matrix)
        np.testing.assert_array_equal(actual, result_matrix)

    def test_XOR(self):
        value1 = ' 000110 110000 001011 101111 111111 000111 000001 110010'
        value_matrix1 = DES.KeyMatrixBuilder(value1, (1, 48)).build().flatten()
        value2 = '011110 100001 010101 010101 011110 100001 010101 010101'
        value_matrix2 = DES.KeyMatrixBuilder(value2, (1, 48)).build().flatten()
        result = ' 011000 010001 011110 111010 100001 100110 010100 100111'.replace(' ', '')
        result_matrix = DES.KeyMatrixBuilder(result, (1, 48)).build().flatten()
        actual = XOR.apply(value_matrix1, value_matrix2)
        np.testing.assert_array_equal(actual, result_matrix)

    def test_transform(self):
        key = '''
                    16   7  20  21
                         29  12  28  17
                          1  15  23  26
                          5  18  31  10
                          2   8  24  14
                         32  27   3   9
                         19  13  30   6
                         22  11   4  25
'''
        print(repr(np.array(key.rsplit()).reshape((8, 4)).astype(int)))

    def test_IP(self):
        value = '0000 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1111'
        value_matrix = DES.KeyMatrixBuilder(value, (1, 64)).build().flatten()
        result = '1100 1100 0000 0000 1100 1100 1111 1111 1111 0000 1010 1010 1111 0000 1010 1010'.replace(' ', '')
        result_matrix = DES.KeyMatrixBuilder(result, (1, 64)).build().flatten()
        actual = IP.apply(value_matrix)
        np.testing.assert_array_equal(actual, result_matrix)

    def test_split(self):
        value = '1100 1100 0000 0000 1100 1100 1111 1111 1111 0000 1010 1010 1111 0000 1010 1010'
        value_matrix = DES.KeyMatrixBuilder(value, (1, 64)).build().flatten()
        result = '1100 1100 0000 0000 1100 1100 1111 1111'
        result_matrix = DES.KeyMatrixBuilder(result, (1, 32)).build().flatten()
        actual = np.split(value_matrix, 2)[0]
        np.testing.assert_array_equal(actual, result_matrix)

    def test_split_r(self):
        value = '1100 1100 0000 0000 1100 1100 1111 1111 1111 0000 1010 1010 1111 0000 1010 1010'
        value_matrix = DES.KeyMatrixBuilder(value, (1, 64)).build().flatten()
        result = '1111 0000 1010 1010 1111 0000 1010 1010'
        result_matrix = DES.KeyMatrixBuilder(result, (1, 32)).build().flatten()
        actual = np.split(value_matrix, 2)[1]
        np.testing.assert_array_equal(actual, result_matrix)

    def test_S(self):
        value = ' 011000 010001 011110 111010 100001 100110 010100 100111'
        value_matrix = DES.KeyMatrixBuilder(value, (1, 48)).build().flatten()
        result = ' 0101 1100 1000 0010 1011 0101 1001 0111'
        result_matrix = DES.KeyMatrixBuilder(result, (1, 32)).build().flatten()
        actual = SSub.apply(value_matrix).flatten()
        np.testing.assert_array_equal(actual, result_matrix)

    def test_permutation(self):
        value = '0101 1100 1000 0010 1011 0101 1001 0111'
        value_matrix = DES.KeyMatrixBuilder(value, (1, 32)).build().flatten()
        result = '0010 0011 0100 1010 1010 1001 1011 1011'
        result_matrix = DES.KeyMatrixBuilder(result, (1, 32)).build().flatten()
        actual = PermutationP.apply(value_matrix).flatten()
        np.testing.assert_array_equal(actual, result_matrix)
