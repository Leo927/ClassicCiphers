from unittest import TestCase
import sys
import os
from playfair import PlayFairCipher


class TestPlayFairCipher(TestCase):

    def alpha_to_num(self, alpha):
        return ord(alpha.lower()) - ord('a') + 1

    def test_key_matrix_is_correct(self):
        print(self.alpha_to_num('A'))
        print(__file__)
