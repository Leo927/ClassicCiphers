import utils
from utils.numtoa import NumToAlphabetConverter
import numpy as np


class Hill:
    def __init__(self, key, decrypt_key):
        self.__key = key
        self.__decrypt_key = decrypt_key

    def encrypt(self, plain_text: str) -> str:
        input_array = self.calculate_input_array(plain_text)
        print(f'''     C({plain_text})=''')
        print(r'''    (\begin{bmatrix}
    17 & 17 & 5 \\
    21 & 18 & 21 \\
    2 & 2 & 19
    \end{bmatrix}''')
        print(f'''    \\begin{{bmatrix}}
    {input_array[0]}\\\\
    {input_array[1]}\\\\
    {input_array[2]}
    \\end{{bmatrix}})mod 26''')
        print(f'''=    \\begin{{bmatrix}}
        (17*{input_array[0]}+17*{input_array[1]}+5*{input_array[2]})mod 26 \\\\
        (21*{input_array[0]}+18*{input_array[1]}+21*{input_array[2]})mod 26 \\\\
        (2*{input_array[0]}+2*{input_array[1]}+19*{input_array[2]})mod 26
    \\end{{bmatrix}}''')
        result = np.mod(np.matmul(self.__key, input_array), 26)
        print(f'''    =
    \\begin{{bmatrix}}
        {result[0]} \\\\
        {result[1]} \\\\
        {result[2]}
    \\end{{bmatrix}}''')
        alphabet_list = list(NumToAlphabetConverter.num_to_alphabet_list(*result))
        result_alphabets = ''.join(alphabet_list).upper()
        print(f'''={result_alphabets}''')
        return result_alphabets

    def decrypt(self, plain_text: str) -> str:
        input_array = self.calculate_input_array(plain_text)
        print(f'''     P({plain_text})=''')
        print(r'''    (\begin{bmatrix}
    4 & 9 & 15 \\
    15 & 17 & 6 \\
    24 & 0 & 17
    \end{bmatrix}''')
        print(f'''    \\begin{{bmatrix}}
    {input_array[0]}\\\\
    {input_array[1]}\\\\
    {input_array[2]}
    \\end{{bmatrix}})mod 26''')
        print(f'''=    \\begin{{bmatrix}}
        (4*{input_array[0]}+9*{input_array[1]}+15*{input_array[2]})mod 26 \\\\
        (15*{input_array[0]}+17*{input_array[1]}+6*{input_array[2]})mod 26 \\\\
        (24*{input_array[0]}+0*{input_array[1]}+17*{input_array[2]})mod 26
    \\end{{bmatrix}}''')
        result = np.mod(np.matmul(self.__decrypt_key, input_array), 26)
        print(f'''    =
    \\begin{{bmatrix}}
        {result[0]} \\\\
        {result[1]} \\\\
        {result[2]}
    \\end{{bmatrix}}''')
        alphabet_list = list(NumToAlphabetConverter.num_to_alphabet_list(*result))
        result_alphabets = ''.join(alphabet_list).upper()
        print(f'''={result_alphabets}''')
        return result_alphabets

    @staticmethod
    def calculate_input_array(plain_text):
        return np.array([i for i in NumToAlphabetConverter.alphabet_to_num_list(*plain_text)])
