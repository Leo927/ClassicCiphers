import math
from typing import Dict, List

import numpy as np

from utils.latexconvert import LatexConverter
from utils.matrixindexer import MatrixIndexer


class DES:
    __sub_key: Dict[int, np.ndarray] = dict()
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

    PC1 = np.array([
        [57, 49, 41, 33, 25, 17, 9],
        [1, 58, 50, 42, 34, 26, 18],
        [10, 2, 59, 51, 43, 35, 27],
        [19, 11, 3, 60, 52, 44, 36],
        [63, 55, 47, 39, 31, 23, 15],
        [7, 62, 54, 46, 38, 30, 22],
        [14, 6, 61, 53, 45, 37, 29],
        [21, 13, 5, 28, 20, 12, 4]
    ])

    PC2 = np.array([
        [14, 17, 11, 24, 1, 5, 3, 28],
        [15, 6, 21, 10, 23, 19, 12, 4],
        [26, 8, 16, 7, 27, 20, 13, 2],
        [41, 52, 31, 37, 47, 55, 30, 40],
        [51, 45, 33, 48, 44, 49, 39, 56],
        [34, 53, 46, 42, 50, 36, 29, 32]
    ])

    correct_PC2 = np.array([[14, 17, 11, 24, 1, 5],
                            [3, 28, 15, 6, 21, 10],
                            [23, 19, 12, 4, 26, 8],
                            [16, 7, 27, 20, 13, 2],
                            [41, 52, 31, 37, 47, 55],
                            [30, 40, 51, 45, 33, 48],
                            [44, 49, 39, 56, 34, 53],
                            [46, 42, 50, 36, 29, 32]])

    S = {1: np.array([[1, 4, 4, 1, 3, 1, 2, 1, 5, 1, 1, 8, 3, 1, 0, 6, 1, 2, 5, 9, 0, 7],
                      [0, 1, 5, 7, 4, 1, 4, 2, 1, 3, 1, 1, 0, 6, 1, 2, 1, 1, 9, 5, 3, 8],
                      [4, 1, 1, 4, 8, 1, 3, 6, 2, 1, 1, 1, 5, 1, 2, 9, 7, 3, 1, 0, 5, 0],
                      [1, 5, 1, 2, 8, 2, 4, 9, 1, 7, 5, 1, 1, 3, 1, 4, 1, 0, 0, 6, 1, 3]]),
         2: np.array([[1, 5, 1, 8, 1, 4, 6, 1, 1, 3, 4, 9, 7, 2, 1, 3, 1, 2, 0, 5, 1, 0],
                      [3, 1, 3, 4, 7, 1, 5, 2, 8, 1, 4, 1, 2, 0, 1, 1, 0, 6, 9, 1, 1, 5],
                      [0, 1, 4, 7, 1, 1, 1, 0, 4, 1, 3, 1, 5, 8, 1, 2, 6, 9, 3, 2, 1, 5],
                      [1, 3, 8, 1, 0, 1, 3, 1, 5, 4, 2, 1, 1, 6, 7, 1, 2, 0, 5, 1, 4, 9]]),
         3: np.array([[1, 0, 0, 9, 1, 4, 6, 3, 1, 5, 5, 1, 1, 3, 1, 2, 7, 1, 1, 4, 2, 8],
                      [1, 3, 7, 0, 9, 3, 4, 6, 1, 0, 2, 8, 5, 1, 4, 1, 2, 1, 1, 1, 5, 1],
                      [1, 3, 6, 4, 9, 8, 1, 5, 3, 0, 1, 1, 1, 2, 1, 2, 5, 1, 0, 1, 4, 7],
                      [1, 1, 0, 1, 3, 0, 6, 9, 8, 7, 4, 1, 5, 1, 4, 3, 1, 1, 5, 2, 1, 2]]),
         4: np.array([[7, 1, 3, 1, 4, 3, 0, 6, 9, 1, 0, 1, 2, 8, 5, 1, 1, 1, 2, 4, 1, 5],
                      [1, 3, 8, 1, 1, 5, 6, 1, 5, 0, 3, 4, 7, 2, 1, 2, 1, 1, 0, 1, 4, 9],
                      [1, 0, 6, 9, 0, 1, 2, 1, 1, 7, 1, 3, 1, 5, 1, 3, 1, 4, 5, 2, 8, 4],
                      [3, 1, 5, 0, 6, 1, 0, 1, 1, 3, 8, 9, 4, 5, 1, 1, 1, 2, 7, 2, 1, 4]]),
         5: np.array([[2, 1, 2, 4, 1, 7, 1, 0, 1, 1, 6, 8, 5, 3, 1, 5, 1, 3, 0, 1, 4, 9],
                      [1, 4, 1, 1, 2, 1, 2, 4, 7, 1, 3, 1, 5, 0, 1, 5, 1, 0, 3, 9, 8, 6],
                      [4, 2, 1, 1, 1, 1, 0, 1, 3, 7, 8, 1, 5, 9, 1, 2, 5, 6, 3, 0, 1, 4],
                      [1, 1, 8, 1, 2, 7, 1, 1, 4, 2, 1, 3, 6, 1, 5, 0, 9, 1, 0, 4, 5, 3]]),
         6: np.array([[1, 2, 1, 1, 0, 1, 5, 9, 2, 6, 8, 0, 1, 3, 3, 4, 1, 4, 7, 5, 1, 1],
                      [1, 0, 1, 5, 4, 2, 7, 1, 2, 9, 5, 6, 1, 1, 3, 1, 4, 0, 1, 1, 3, 8],
                      [9, 1, 4, 1, 5, 5, 2, 8, 1, 2, 3, 7, 0, 4, 1, 0, 1, 1, 3, 1, 1, 6],
                      [4, 3, 2, 1, 2, 9, 5, 1, 5, 1, 0, 1, 1, 1, 4, 1, 7, 6, 0, 8, 1, 3]]),
         7: np.array([[4, 1, 1, 2, 1, 4, 1, 5, 0, 8, 1, 3, 3, 1, 2, 9, 7, 5, 1, 0, 6, 1],
                      [1, 3, 0, 1, 1, 7, 4, 9, 1, 1, 0, 1, 4, 3, 5, 1, 2, 2, 1, 5, 8, 6],
                      [1, 4, 1, 1, 1, 3, 1, 2, 3, 7, 1, 4, 1, 0, 1, 5, 6, 8, 0, 5, 9, 2],
                      [6, 1, 1, 1, 3, 8, 1, 4, 1, 0, 7, 9, 5, 0, 1, 5, 1, 4, 2, 3, 1, 2]]),
         8: np.array([[1, 3, 2, 8, 4, 6, 1, 5, 1, 1, 1, 1, 0, 9, 3, 1, 4, 5, 0, 1, 2, 7],
                      [1, 1, 5, 1, 3, 8, 1, 0, 3, 7, 4, 1, 2, 5, 6, 1, 1, 0, 1, 4, 9, 2],
                      [7, 1, 1, 4, 1, 9, 1, 2, 1, 4, 2, 0, 6, 1, 0, 1, 3, 1, 5, 3, 5, 8],
                      [2, 1, 1, 4, 7, 4, 1, 0, 8, 1, 3, 1, 5, 1, 2, 9, 0, 3, 5, 6, 1, 1]])
         }

    def __init__(self, key: np.ndarray, size=64):
        self.__size = size
        self.__round = 1
        self.__key_matrix = key
        self.__value = self.__key_matrix
        print(f'initial key is: \n{LatexConverter.convert_to_latex(self.__key_matrix)}')

    def generate_sub_key(self):
        print(f'round {self.__round}')
        if self.__round == 1:
            self.__key_matrix = MatrixIndexer(self.PC1).convert(self.__key_matrix)
            print(f'do PC 1\n{LatexConverter.convert_to_latex(self.__key_matrix)}')
        self.__key_matrix = LeftShifter(self.__round).shift(self.__key_matrix)

        after_pc2 = MatrixIndexer(self.correct_PC2).convert(self.__key_matrix)
        print(f'do PC 2\n{LatexConverter.convert_to_latex(after_pc2)}')
        self.__sub_key[self.__round] = after_pc2
        return after_pc2

    class KeyMatrixBuilder:
        __key_matrix: np.ndarray

        def __init__(self, key, shape=None):
            self.__key = key
            if shape is None:
                key_len = len(self.__key)
                width = int(math.sqrt(key_len))
                self.__shape = (width, width)
            else:
                self.__shape = shape

        def build(self):
            full_key_matrix = np.array(list(map(lambda i: int(i), list(self.__key)))).reshape(
                self.__shape)
            self.__key_matrix = full_key_matrix
            return self.__key_matrix

    def get_l_r_key(self):
        return self.split_matrix(self.__key_matrix)

    def get_l_r_value(self):
        return self.split_matrix(self.__value)

    @staticmethod
    def split_matrix(matrix):
        return np.split(matrix, 2)

    def do_round(self):
        if self.__round == 1:
            self.__value = MatrixIndexer(self.initial_permutation_matrix).convert(self.__value)
            print(f'IP(P_0)=\n{LatexConverter.convert_to_latex(self.__value)}')
        current_round = self.__round - 1
        left, right = self.get_l_r_value()
        print(f'L_{current_round}=\n{LatexConverter.convert_to_latex(left)}')
        print(f'R_{current_round}=\n{LatexConverter.convert_to_latex(right)}')

        e_right = MatrixIndexer(self.expansion_matrix).convert(right)
        print(f'E[R_{current_round}]=\n{LatexConverter.convert_to_latex(e_right)}')

        xor = np.logical_xor(e_right, self.__sub_key[self.__round]).astype(int)
        print(f'E[R_{current_round}]\\oplus K_{self.__round}=\n{LatexConverter.convert_to_latex(xor)}')

        after_s_substitute = self.substitute_s(xor)

    def increment_round(self):
        self.__round += 1

    def substitute_s(self, xor):
        arrays = [np.split(xor, 8)][0]
        result_matrices_buffer = []
        for i in range(8):
            six_bits = arrays[i]
            print(six_bits)
            print(f'row: {i + 1}. Six bits:\n{LatexConverter.convert_to_latex(six_bits)}')
            row_index, col_index = self.get_substitution_indexes(six_bits)
            current_s = self.S[i + 1]
            value = current_s[row_index, col_index]
            value_matrix = self.to_binary(value)
            print(f'value is {value_matrix}')
            result_matrices_buffer.append(value_matrix)
        result_matrix = np.array(result_matrices_buffer)
        print(f'S(E(R)\\oplusK=\n{LatexConverter.convert_to_latex(result_matrix)}')
        return result_matrix

    @staticmethod
    def get_substitution_indexes(six_bits):
        six_bits = six_bits.flatten()
        col = sum(six_bits[1:5])
        row = sum(six_bits) - col
        return row, col

    @staticmethod
    def to_binary(value) -> List[int]:
        no_padded = [int(x) for x in list('{0:0b}'.format(value))]
        pad_to = 4
        padded = [0] * (pad_to - len(no_padded)) + no_padded
        return np.array(padded)


class LeftShifter:
    shift_schedule = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]

    def __init__(self, num_round):
        self.round = num_round

    def shift(self, value: np.ndarray):
        num_left_shift = self.shift_schedule[self.round - 1]
        l, r = np.split(value.flatten(), 2)
        shift_l = np.roll(l, -num_left_shift)
        shift_r = np.roll(r, -num_left_shift)
        print(f'left shifting {num_left_shift}')
        result = np.concatenate([shift_l, shift_r])
        print(f'left shift result:\n{LatexConverter.convert_to_latex(result.reshape(8, 7))}')
        return result
