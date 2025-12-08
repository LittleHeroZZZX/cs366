import json
import os
from collections.abc import Iterable, Iterator
from multiprocessing import Pool
from pathlib import Path
from typing import Self

import numpy as np
import regex as re
from cachetools import LRUCache
from tqdm import tqdm

from ._types import PreToken, Vocab
from .pre_tokenization import MultiProcessPreTokenizer
from .utils import find_chunk_boundaries, make_file_str_iter


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
        self.ranks = {pair: i for i, pair in enumerate(merges)}

        self.cache = LRUCache(maxsize=10000)
        self.pre_tokenizer = MultiProcessPreTokenizer()

        assert len(self.vocab) == len(self.token_to_id), "Vocab contains duplicate tokens."

    def encode(self, text: str | bytes) -> list[int]:
        byte_text = text.encode("utf-8") if isinstance(text, str) else text
        id_list = []
        for pre_token in self.pre_tokenizer.pre_tokenize(byte_text, self.special_tokens):
            id_list.extend(self._encode_one_token(pre_token))
        return id_list

    def encode_batch(
        self, texts: Iterable[str | bytes], num_workers: int | None = None, save_file: str | None = None
    ) -> list[list[int]]:
        if not texts:
            return []
        if num_workers is None:
            cpu_count = os.cpu_count() or 10
            num_workers = max(1, cpu_count - 1)
        results = []

        if save_file is None:
            with Pool(processes=num_workers) as pool:
                for chunk_ids in tqdm(
                    pool.imap(self.encode, texts, chunksize=1), desc="Encoding", total=len(list(texts))
                ):
                    results.append(chunk_ids)
        else:
            with open(save_file, "wb") as f_out, Pool(processes=num_workers) as pool:
                for chunk_ids in tqdm(
                    pool.imap(self.encode, texts, chunksize=1), desc="Encoding", total=len(list(texts))
                ):
                    res = np.array(chunk_ids, dtype=np.uint16)
                    res.tofile(f_out)

        return results

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            byte_text = text.encode("utf-8")
            for pre_token in self.pre_tokenizer.pre_tokenize(byte_text, self.special_tokens):
                yield from self._encode_one_token(pre_token)

    def _encode_one_token(self, token: PreToken) -> list[int]:
        if token in self.cache:
            return self.cache[token]

        if token in self.token_to_id:
            return [self.token_to_id[token]]

        word = [token[i : i + 1] for i in range(len(token))]

        while len(word) > 1:
            min_rank = float("inf")
            min_pair = None

            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                rank = self.ranks.get(pair, float("inf"))
                if rank < min_rank:
                    min_rank = rank
                    min_pair = pair

            if min_pair is None or min_rank == float("inf"):
                break

            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == min_pair[0] and word[i + 1] == min_pair[1]:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word

        ids_list = [self.token_to_id[token_bytes] for token_bytes in word]
        self.cache[token] = ids_list
        return ids_list

    def decode(self, token_ids: list[int]) -> str:
        byte_list = [self.vocab[token_id] for token_id in token_ids]
        return b"".join(byte_list).decode("utf-8", errors="replace")

    @classmethod
    def from_file(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> Self:
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

    def encode_file(
        self, file_path: str, split_token: str = "<|endoftext|>", save_file: str | None = None
    ) -> list[int]:
        boundries = find_chunk_boundaries(
            file_path=file_path,
            desired_num_chunks=(os.cpu_count() or 8) * 100,
            split_special_token=split_token.encode(),
            desize_bytes=1024 * 1024,
        )
        file_iter = make_file_str_iter(file_path=file_path, boundaries=boundries, return_bytes=True)
        ret = self.encode_batch(file_iter, save_file=save_file)
        return [list_id for sublist in ret for list_id in sublist]

    @classmethod
    def from_my_save(cls, save_dir: str | Path, special_tokens: list[str] | None = None) -> Self:
        """
        Build a Tokenizer from custom saved vocab and merges files.

        Args:
            save_dir (str): Directory containing vocab.txt and merges.txt.
            special_tokens (list[str]): Special tokens. Defaults to ["<|endoftext|>"].

        Returns:
            Self: An instance of Tokenizer.
        """

        if special_tokens is None:
            special_tokens = ["<|endoftext|>"]
        directory = Path(save_dir)

        with open(directory / "vocab.json", encoding="utf-8") as f:
            vocab_str = json.load(f)

        vocab = {}
        vocab_size = 0
        for token_str, token_id in vocab_str.items():
            token_bytes = token_str.encode("latin-1")
            vocab[token_id] = token_bytes
        vocab_size = vocab_size

        merges = []
        merges_file = directory / "merges.txt"
        if merges_file.exists():
            with open(merges_file, encoding="utf-8") as f:
                for line in f:
                    regex = r"^(.*) (.*)$"
                    match = re.match(regex, line.rstrip())
                    if match:
                        token1 = match.group(1).encode("latin-1")
                        token2 = match.group(2).encode("latin-1")
                        merges.append((token1, token2))
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
