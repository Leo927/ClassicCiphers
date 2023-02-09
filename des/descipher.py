import math

import numpy as np


class DES:
    initial_permutation_matrix = np.array([
        [58, 50, 42, 34, 26, 18, 10, 2],
        [60, 52, 44, 36, 28, 20, 12, 4],
        [62, 54, 46, 38, 30, 22, 14, 6],
        [64, 56, 48, 40, 32, 24, 16, 8],
        [57, 49, 41, 33, 25, 17, 9, 1],
        [59, 51, 43, 35, 27, 19, 11, 3],
        [61, 53, 45, 37, 29, 21, 13, 5],
        [63, 55, 47, 39, 31, 23, 15, 7]
    ])

    inverse_initial_permutation_matrix = np.array([
        [40, 8, 48, 16, 56, 24, 64, 32],
        [39, 7, 47, 15, 55, 23, 63, 31],
        [38, 5, 45, 13, 53, 21, 61, 29],
        [36, 4, 44, 12, 52, 20, 60, 28],
        [35, 3, 43, 11, 51, 19, 59, 27],
        [34, 2, 42, 10, 50, 18, 58, 26],
        [33, 1, 41, 9, 49, 17, 57, 25]
    ])

    expansion_matrix = np.array([
        [32, 1, 2, 3, 4, 5],
        [4, 5, 6, 7, 8, 9],
        [8, 9, 10, 11, 12, 13],
        [12, 13, 14, 15, 16, 17],
        [16, 17, 18, 19, 20, 21],
        [20, 21, 22, 23, 24, 25],
        [24, 25, 26, 27, 28, 29],
        [28, 29, 30, 31, 32, 1]
    ])

    P1 = np.array([
        [57, 49, 41, 33, 25, 17, 9],
        [1, 58, 50, 42, 34, 26, 18],
        [10, 2, 59, 51, 43, 35, 27],
        [19, 11, 3, 60, 52, 44, 36]
    ])

    P2 = np.array([
        [63, 55, 47, 39, 31, 23, 15],
        [7, 62, 54, 46, 38, 30, 22],
        [14, 6, 61, 53, 45, 37, 29],
        [21, 13, 5, 28, 20, 12, 4]
    ])

    def __init__(self, key: np.ndarray, size=64):
        self.__key = key
        self.__size = size
        self.__key_matrix = self.KeyMatrixBuilder(key, size).build()
        self.initial_permutation()

    def initial_permutation(self):
        self.__C = self.MatrixIndexer(self.P1).convert(self.__key_matrix)
        self.__D = self.MatrixIndexer(self.P2).convert(self.__key_matrix)
        print(f'C=\n{self.__C}')
        print(f'D=\n{self.__D}')

    def generate_sub_key(self):
        pass

    class KeyMatrixBuilder:
        __key_matrix: np.ndarray

        def __init__(self, key, size):
            self.__key = key
            self.__size = size

        def build(self):
            key_matrix_width = int(math.sqrt(self.__size))
            full_key_matrix = np.array(list(map(lambda i: int(i), list(self.__key)))).reshape(
                (key_matrix_width, key_matrix_width))
            self.__key_matrix = full_key_matrix
            return self.__key_matrix

    class MatrixIndexer:
        def __init__(self, key: np.ndarray):
            self.__key = key

        def convert(self, value: np.ndarray):
            result = np.zeros(self.__key.shape)
            for row in range(self.__key.shape[0]):
                for col in range(self.__key.shape[1]):
                    result[row, col] = self.__index_value(value, row, col)
            return result

        def __index_value(self, value, row, col):
            index = self.__key[row, col] - 1
            flat_value = value.flatten()
            return flat_value[index]

    def get_l_r_key(self):
        shape = self.__key_matrix.shape
        return (self.__key_matrix[:(shape[0] // 2), ],
                self.__key_matrix[(shape[0] // 2):, ])
