import numpy as np


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
