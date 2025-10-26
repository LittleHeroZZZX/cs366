# native implementation, single process, no optimization
from ..typings import vocab_T, merges_T, word_count_T
from .utils import split_text_by_special_token
import regex as re
from collections import defaultdict


def perform_pretokenization(str_bytes: bytes, special_tokens: list[bytes]) -> word_count_T:
    """
    Count the occurence of each word on the given str_bytes
    """
    word_count: word_count_T = defaultdict(int)
    splited_bytes = split_text_by_special_token(str_bytes, special_tokens)
    pattern = re.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    for text in splited_bytes:
        for token in re.finditer(pattern, text):
            word_count[token.group()] += 1
    return word_count


def tokenize(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[vocab_T, merges_T]:
    vocab: vocab_T = {}
    merges: merges_T = []

    return vocab, merges
