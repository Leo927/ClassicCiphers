from __future__ import annotations
from typing import Tuple


class PlayFairCipher:
    def __init__(self, key: str):
        self.__key_matrix = self.KeyMatrixBuilder(key).build()

    class KeyMatrix:
        def convert(self, first_letter, second_letter) -> Tuple[str, str]:
            pass

    class KeyMatrixBuilder:
        def __init__(self, key: str):
            self.__key = list(key)

        def build(self) -> dict:
            self.__calculate_full_key()
            self.__calculate_key_matrix()
            return self.__key_matrix

        def __calculate_full_key(self) -> list:
            self.__full_key = self.__key
            remaining_letters = [k for k in self.__get_remaining_letters() if k not in self.__key]
            for k in remaining_letters:
                self.__full_key.append(k)
            return self.__full_key

        def __get_remaining_letters(self):
            for num in range(26):
                if num not in self.__key and num != ord('j'):
                    if num == ord('i'):
                        yield 'i/j'
                    else:
                        yield self.__num_to_alphabet(num)

        @staticmethod
        def __num_to_alphabet(i):
            return chr(ord('a') + i)

        def __calculate_key_matrix(self) -> dict:
            index = 0
            self.__key_matrix = dict()
            for row in range(5):
                for col in range(5):
                    self.__key_matrix[(row, col)] = self.__full_key[index]
                    index += 1
            return self.__key_matrix


