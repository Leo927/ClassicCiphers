from typing import Iterable


class NumToAlphabetConverter:
    @classmethod
    def num_to_alphabet(cls, num: int) -> str:
        return chr(ord('a') + num)

    @classmethod
    def num_to_alphabet_list(cls, *num: int) -> Iterable[str]:
        for n in num:
            yield cls.num_to_alphabet(n)

    @classmethod
    def alphabet_to_num(cls, alphabet: str) -> int:
        return ord(alphabet.lower()) - ord('a')

    @classmethod
    def alphabet_to_num_list(cls, *alphabet: str) -> Iterable[int]:
        return (cls.alphabet_to_num(a) for a in alphabet)
