import math
from typing import Dict, List

import numpy as np

from utils.latexconvert import LatexConverter
from utils.matrixindexer import MatrixIndexer


class DES:
    __sub_key: Dict[int, np.ndarray] = dict()

    inverse_initial_permutation_matrix = np.array([
        [40, 8, 48, 16, 56, 24, 64, 32],
        [39, 7, 47, 15, 55, 23, 63, 31],
        [38, 5, 45, 13, 53, 21, 61, 29],
        [36, 4, 44, 12, 52, 20, 60, 28],
        [35, 3, 43, 11, 51, 19, 59, 27],
        [34, 2, 42, 10, 50, 18, 58, 26],
        [33, 1, 41, 9, 49, 17, 57, 25]
    ])

    correct_PC2 = np.array([[14, 17, 11, 24, 1, 5],
                            [3, 28, 15, 6, 21, 10],
                            [23, 19, 12, 4, 26, 8],
                            [16, 7, 27, 20, 13, 2],
                            [41, 52, 31, 37, 47, 55],
                            [30, 40, 51, 45, 33, 48],
                            [44, 49, 39, 56, 34, 53],
                            [46, 42, 50, 36, 29, 32]])

    def __init__(self, key: np.ndarray, value: np.ndarray, size=64):
        self.__size = size
        self.__round = 1
        self.__key_matrix = key.flatten()
        self.__value = value
        self.__left = {}
        self.__right = {}
        print(f'initial key is: \n{LatexConverter.convert_to_latex(self.__key_matrix)}')

    def generate_sub_key(self):
        print(f'round {self.__round}')
        if self.__round == 1:
            self.__key_matrix = PC1.apply(self.__key_matrix)
            print(f'do PC 1\n{LatexConverter.convert_to_latex(self.__key_matrix)}')
        self.__key_matrix = LeftShifter(self.__round).shift(self.__key_matrix)

        after_pc2 = MatrixIndexer(self.correct_PC2).convert(self.__key_matrix)
        print(f'do PC 2\n{LatexConverter.convert_to_latex(after_pc2)}')
        self.__sub_key[self.__round] = after_pc2.flatten()
        return after_pc2

    class KeyMatrixBuilder:
        __key_matrix: np.ndarray

        def __init__(self, key, shape=None):
            self.__key = key.replace(' ', '')
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

    @staticmethod
    def split_matrix(matrix):
        return np.split(matrix, 2)

    def do_round(self):
        next_round = self.__round
        current_round = next_round - 1
        if next_round == 1:
            self.__value = IP.apply(self.__value)
            print(f'IP(P_0)=\n{self.__value}')
            self.__left[current_round], self.__right[current_round] = np.split(self.__value, 2)

        print(f'L_{current_round}=\n{self.__left[current_round]}')
        print(f'R_{current_round}=\n{self.__right[current_round]}')

        self.calculate_next_left(current_round, next_round)

        self.calculate_next_right(current_round, next_round)

        return np.concatenate([self.__left[next_round], self.__right[next_round]])

    def calculate_next_right(self, current_round, next_round):
        e_right = Expansion.apply(self.__right[current_round])
        print(f'E[R_{current_round}]=\n{LatexConverter.convert_to_latex(e_right)}')
        xor = np.logical_xor(e_right, self.__sub_key[next_round]).astype(int)
        print(f'E[R_{current_round}]\\oplus K_{next_round}=\n{LatexConverter.convert_to_latex(xor)}')
        after_s_substitute = SSub.apply(xor)
        print(
            f'S(E[R_{current_round}]\\oplus K_{next_round})=\n{LatexConverter.convert_to_latex(after_s_substitute)}')
        f = PermutationP.apply(after_s_substitute)
        print(f'f(R_{current_round}, K_{next_round}) = {LatexConverter.convert_to_latex(f)}')
        self.__right[next_round] = XOR.apply(self.__left[current_round], f)
        print(
            f'''R_{next_round}=L_{current_round} \\oplus f(R_{current_round},K_{next_round}) =
{LatexConverter.convert_to_latex(self.__left[current_round])} 
            \\oplus 
{LatexConverter.convert_to_latex(f)}''')

    def calculate_next_left(self, current_round, next_round):
        self.__left[next_round] = self.__left[current_round]
        print(f'''L_{next_round}=R_{current_round}={LatexConverter.convert_to_latex(self.__left[next_round])}''')

    def increment_round(self):
        self.__round += 1


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


class PC2:
    matrix = np.array([
        [14, 17, 11, 24, 1, 5, 3, 28],
        [15, 6, 21, 10, 23, 19, 12, 4],
        [26, 8, 16, 7, 27, 20, 13, 2],
        [41, 52, 31, 37, 47, 55, 30, 40],
        [51, 45, 33, 48, 44, 49, 39, 56],
        [34, 53, 46, 42, 50, 36, 29, 32]
    ])

    @classmethod
    def apply(cls, value):
        return MatrixIndexer(cls.matrix).convert(value).flatten()


class PC1:
    matrix = np.array([
        [57, 49, 41, 33, 25, 17, 9],
        [1, 58, 50, 42, 34, 26, 18],
        [10, 2, 59, 51, 43, 35, 27],
        [19, 11, 3, 60, 52, 44, 36],
        [63, 55, 47, 39, 31, 23, 15],
        [7, 62, 54, 46, 38, 30, 22],
        [14, 6, 61, 53, 45, 37, 29],
        [21, 13, 5, 28, 20, 12, 4]
    ])

    @classmethod
    def apply(cls, value):
        return MatrixIndexer(cls.matrix).convert(value).flatten()


class Expansion:
    matrix = np.array([
        [32, 1, 2, 3, 4, 5],
        [4, 5, 6, 7, 8, 9],
        [8, 9, 10, 11, 12, 13],
        [12, 13, 14, 15, 16, 17],
        [16, 17, 18, 19, 20, 21],
        [20, 21, 22, 23, 24, 25],
        [24, 25, 26, 27, 28, 29],
        [28, 29, 30, 31, 32, 1]
    ])

    @classmethod
    def apply(cls, value):
        return MatrixIndexer(cls.matrix).convert(value).flatten()


class XOR:
    @staticmethod
    def apply(value1, value2):
        return np.logical_xor(value1, value2)


class IP:
    matrix = np.array([[58, 50, 42, 34, 26, 18, 10, 2],
                       [60, 52, 44, 36, 28, 20, 12, 4],
                       [62, 54, 46, 38, 30, 22, 14, 6],
                       [64, 56, 48, 40, 32, 24, 16, 8],
                       [57, 49, 41, 33, 25, 17, 9, 1],
                       [59, 51, 43, 35, 27, 19, 11, 3],
                       [61, 53, 45, 37, 29, 21, 13, 5],
                       [63, 55, 47, 39, 31, 23, 15, 7]])

    @classmethod
    def apply(cls, value):
        return MatrixIndexer(cls.matrix).convert(value).flatten()


class SSub:
    S = {1: np.array([[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
                      [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
                      [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
                      [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13]]),
         2: np.array([[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
                      [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
                      [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
                      [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9]]),
         3: np.array([[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
                      [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
                      [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
                      [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12]]),
         4: np.array([[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
                      [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
                      [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
                      [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14]]),
         5: np.array([[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
                      [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
                      [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
                      [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3]]),
         6: np.array([[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
                      [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
                      [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
                      [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13]]),
         7: np.array([[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
                      [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
                      [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
                      [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12]]),
         8: np.array([[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
                      [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
                      [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
                      [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11]])
         }

    @classmethod
    def apply(cls, xor):
        arrays = np.split(xor, 8)
        result_matrices_buffer = []
        for i in range(8):
            six_bits = arrays[i]
            print(six_bits)
            print(f'row: {i + 1}. Six bits:\n{LatexConverter.convert_to_latex(six_bits)}')
            row_index, col_index = cls.get_substitution_indexes(six_bits)
            current_s = cls.S[i + 1]
            value = current_s[row_index, col_index]
            print(f'value is {value}')
            value_matrix = cls.to_binary(value)
            print(f'value is {value_matrix}')
            result_matrices_buffer.append(value_matrix)
        result_matrix = np.array(result_matrices_buffer)
        print(f'S(E(R)\\oplusK=\n{LatexConverter.convert_to_latex(result_matrix)}')
        return result_matrix

    @staticmethod
    def get_substitution_indexes(six_bits):
        six_bits = six_bits.flatten()
        col = int(''.join(map(str, six_bits[1:5])), 2)
        row = int(''.join(map(str, [six_bits[0], six_bits[-1]])), 2)
        print(f'index: ({row}, {col})')
        return row, col

    @staticmethod
    def to_binary(value) -> List[int]:
        no_padded = [int(x) for x in list('{0:0b}'.format(value))]
        pad_to = 4
        padded = [0] * (pad_to - len(no_padded)) + no_padded
        return np.array(padded)


class PermutationP:
    matrix = np.array([[16, 7, 20, 21],
                       [29, 12, 28, 17],
                       [1, 15, 23, 26],
                       [5, 18, 31, 10],
                       [2, 8, 24, 14],
                       [32, 27, 3, 9],
                       [19, 13, 30, 6],
                       [22, 11, 4, 25]])

    @classmethod
    def apply(cls, value):
        return MatrixIndexer(cls.matrix).convert(value).flatten()
