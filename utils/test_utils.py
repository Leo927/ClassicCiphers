from unittest import TestCase
import os
from numtoa import NumToAlphabetConverter


class TestNumToAlphabetConverter(TestCase):
    def test_a_to_one(self):
        self.assertEqual(NumToAlphabetConverter.alphabet_to_num('a'), 0)
        self.assertEqual(NumToAlphabetConverter.alphabet_to_num('z'), 25)

    def test_one_to_a(self):
        self.assertEqual(NumToAlphabetConverter.num_to_alphabet(0), 'a')
        self.assertEqual(NumToAlphabetConverter.num_to_alphabet(25), 'z')

