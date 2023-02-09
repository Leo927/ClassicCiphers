from unittest import TestCase
from hillcipher import Hill
import numpy as np
from numpy import testing


class TestHill(TestCase):
    def test_encrypt(self):
        key = np.array([[17, 17, 5], [21, 18, 21], [2, 2, 19]])
        hill = Hill(key)
        print(hill.encrypt('TME'))

    def test_decrypt(self):
        key = np.array([[17, 17, 5],
                        [21, 18, 21],
                        [2, 2, 19]])
        decrypt_key = np.array(
            [[4, 9, 15],
             [15, 17, 6],
             [24, 0, 17]])
        hill = Hill(key, decrypt_key)
        cipher_texts = 'GSEB XI CN JBP MWKYD EBCQC'.replace(' ', '')
        chunks, chunk_size = len(cipher_texts), len(cipher_texts)//6
        cipher_texts_chunks = [cipher_texts[i:i+chunk_size] for i in range(0, chunks, chunk_size)]
        for c in cipher_texts_chunks:
            hill.decrypt(c)
            print(r'\\')

    def test_calculate_input_array(self):
        key = np.array([[17, 17, 5], [21, 18, 21], [2, 2, 19]])
        hill = Hill(key)
        testing.assert_array_equal(hill.calculate_input_array('MMT'), np.array([12, 12, 19]))

    def test_encrypt_decrypt_match(self):
        key = np.array([[17, 17, 5],
                        [21, 18, 21],
                        [2, 2, 19]])
        decrypt_key = np.array(
                       [[4, 9, 15],
                        [15, 17, 6],
                        [24, 0, 17]])
        hill = Hill(key, decrypt_key)
        plain_text = 'ACE'
        cipher_text = hill.encrypt(plain_text)
        self.assertEqual(plain_text, hill.decrypt(cipher_text))


