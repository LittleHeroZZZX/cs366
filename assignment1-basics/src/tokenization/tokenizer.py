from collections.abc import Iterable, Iterator
from typing import Self

import regex as re
from cachetools import LRUCache

from .pre_tokenization import MultiProcessPreTokenizer
from .types import PreToken, Vocab


class Tokenizer:
    def __init__(
        self,
        vocab: Vocab,
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = [token.encode() for token in special_tokens] if special_tokens else []

        for token in self.special_tokens:
            if token not in self.vocab.values():
                self.vocab[len(self.vocab)] = token

        self.token_to_id = {token: idx for idx, token in self.vocab.items()}
        self.cache = LRUCache(maxsize=10000)  # TODO: profile different size
        self.pre_tokenizer = MultiProcessPreTokenizer()

        assert len(self.vocab) == len(self.token_to_id), "Vocab contains duplicate tokens."

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> Self:
        """
        Build a Tokenizer from vocab and merges files.

        Args:
            vocab_filepath (str): Vocab file path.
            merges_filepath (str): Merges file path.
            special_tokens (list[str] | None): Special tokens. Defaults to None.

        Returns:
            Self: An instance of Tokenizer.
        """
        vocab: Vocab = {}
        vocal_pattern = rb'"(.*)"'
        idx = 0
        with open(vocab_filepath, "rb") as vf:
            for match in re.finditer(vocal_pattern, vf.read()):
                token = match.group(1)
                vocab[idx] = token
                idx += 1

        merge_pattern = rb"^(.*) (.*)$"
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "rb") as mf:
            for line in mf:
                match = re.match(merge_pattern, line.rstrip())
                if match:
                    token1 = match.group(1)
                    token2 = match.group(2)
                    merges.append((token1, token2))
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        Encode the given text into a list of token IDs.

        Args:
            text (str): The input text to encode.

        Returns:
            list[int]: A list of token IDs.
        """
        byte_text = text.encode("utf-8")
        id_list = []
        for pre_token in self.pre_tokenizer.pre_tokenize(byte_text, self.special_tokens):
            id_list.extend(self._encode_one_token(pre_token))
        return id_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Encode an iterable of strings into an iterator of token IDs.

        Args:
            iterable (Iterable[str]): An iterable of input strings.

        Returns:
            Iterator[int]: An iterator of token IDs.
        """
        for text in iterable:
            byte_text = text.encode("utf-8")
            for pre_token in self.pre_tokenizer.pre_tokenize(byte_text, self.special_tokens):
                yield from self._encode_one_token(pre_token)

    def _encode_one_token(self, token: PreToken) -> list[int]:
        """
        Encode a single token into a list of token IDs using BPE merges.

        Args:
            token (PreToken): The input token to encode.

        Returns:
            list[int]: A list of token IDs.
        """
        if token in self.cache:
            return self.cache[token]
        if token in self.token_to_id:
            return [self.token_to_id[token]]
        ids_list: list[int] = []
        token_state = [bytes([b]) for b in token]
        for merge_pair in self.merges:
            if len(token_state) <= 1:
                break
            pos = 0
            while pos < len(token_state) - 1:
                if (token_state[pos], token_state[pos + 1]) == merge_pair:
                    pair = token_state[pos] + token_state[pos + 1]
                    token_state = [*token_state[:pos], pair, *token_state[pos + 2 :]]
                pos += 1
        for byte_token in token_state:
            ids_list.append(self.token_to_id[byte_token])
        self.cache[token] = ids_list
        return ids_list

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode a list of token IDs back into a string.

        Args:
            token_ids (list[int]): A list of token IDs to decode.

        Returns:
            str: The decoded string.
        """
        byte_list = [self.vocab[token_id] for token_id in token_ids]
        return b"".join(byte_list).decode("utf-8", errors="replace")
