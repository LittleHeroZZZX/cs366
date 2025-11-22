import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Iterator
from logging import basicConfig, getLogger
from multiprocessing import Pool, cpu_count
from typing import BinaryIO

import regex as re
import tqdm

from .types import Chunk, PreTokenCount, Token

logger = getLogger(__name__)
basicConfig(level="INFO")


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: Token,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


class PreTokenizer(ABC):
    @staticmethod
    def _merge_pre_token_counts(*pre_token_counts: PreTokenCount) -> PreTokenCount:
        """Merge multiple PreTokenCount dictionaries into one.

        Returns:
            PreTokenCount: The merged PreTokenCount.
        """
        merged_pre_token_count: PreTokenCount = defaultdict(int)
        for pre_token_count in pre_token_counts:
            for pre_token, count in pre_token_count.items():
                merged_pre_token_count[pre_token] += count
        return merged_pre_token_count

    def _process_chunk(self, chunk: Chunk, special_tokens: list[Token]) -> PreTokenCount:
        """
        Process a single chunk of text and return the pre-token counts.

        Args:
            chunk (Chunk): The chunk of text to process.

        Returns:
            PreTokenCount: A dictionary-like object mapping pre-tokens to their counts.
        """
        pre_token_count: PreTokenCount = defaultdict(int)
        pattern = rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for mini_chunk in re.split(b"|".join([re.escape(token) for token in special_tokens]), chunk):
            for token_match in re.finditer(pattern, mini_chunk):
                pre_token = token_match.group()
                pre_token_count[pre_token] += 1
        return pre_token_count

    def pre_tokenize(self, str_bytes: bytes, special_token_list: list[Token]) -> Iterator[Token]:
        """
        Pre-tokenize the given bytes string.

        Args:
            str_bytes (bytes): The input bytes string to pre-tokenize.
            special_token_list (list[Token]): The list of special tokens.
        Returns:
            Iterator[Token]: An iterator over the pre-tokens.
        """
        # TODO:
        # 0. 特殊字符要通过 split 先分割出来
        # 1. 按照从长到短排序
        special_token_list = sorted(special_token_list, key=len, reverse=True)  # match longer tokens first
        special_token_pattern = (
            b"|".join([re.escape(token) for token in special_token_list]) if special_token_list else b""
        )
        special_token_pattern = b"(" + special_token_pattern + b")"
        pattern = re.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        if special_token_list:
            for mini_chunk in re.splititer(special_token_pattern, str_bytes):
                if mini_chunk in special_token_list:
                    yield mini_chunk
                    continue
                for token_match in re.finditer(pattern, mini_chunk):
                    yield token_match.group()
        else:
            for token_match in re.finditer(pattern, str_bytes):
                yield token_match.group()

    @abstractmethod
    def __call__(self, corpos_path: str, split_special_token: Token, special_tokens: list[Token]) -> PreTokenCount:
        """
        Pre-tokenize the given corpus.

        Args:
            corpos_path (str): Path to the corpus file.
            split_special_token (token): The special token used to split the corpus.
            special_tokens (list[Token]): List of special tokens.

        Returns:
            PreTokenCount: A dictionary-like object mapping pre-tokens to their counts.
        """


class NativePreTokenizer(PreTokenizer):
    def __call__(
        self, corpos_path: str, split_special_token: Token, special_tokens: list[Token], num_chunks: int = 8
    ) -> PreTokenCount:
        pre_token_count: PreTokenCount = defaultdict(int)

        start_time = time.time()
        with open(corpos_path, mode="br") as f:
            file_size = os.path.getsize(corpos_path)
            chunk_boundaries = find_chunk_boundaries(
                file=f,
                desired_num_chunks=num_chunks,
                split_special_token=split_special_token,
            )

            for i in tqdm.tqdm(range(len(chunk_boundaries) - 1), desc="Pre-tokenizing corpus"):
                start = chunk_boundaries[i]
                end = chunk_boundaries[i + 1]
                f.seek(start)
                chunk = f.read(end - start)
                chunk_pre_token_count = self._process_chunk(chunk, special_tokens)
                for pre_token, count in chunk_pre_token_count.items():
                    pre_token_count[pre_token] += count
        end_time = time.time()
        logger.info(
            "Takes %.2f seconds to pre-tokenize the corpus file %s, speed: %.2f bytes/second",
            end_time - start_time,
            corpos_path,
            file_size / (end_time - start_time),
        )
        return pre_token_count


class MultiProcessPreTokenizer(PreTokenizer):
    def _process_chunk_with_boundry(
        self, corpos_path: str, start: int, end: int, special_tokens: list[Token]
    ) -> PreTokenCount:
        with open(corpos_path, mode="br") as f:
            f.seek(start)
            chunk = f.read(end - start)
            pre_token_count = self._process_chunk(chunk, special_tokens)
        return pre_token_count

    def __call__(self, corpos_path: str, split_special_token: Token, special_tokens: list[Token]) -> PreTokenCount:
        pre_token_count: PreTokenCount = defaultdict(int)

        start_time = time.time()
        with open(corpos_path, mode="br") as f:
            file_size = os.path.getsize(corpos_path)
            num_cpus = cpu_count()
            chunk_boundaries = find_chunk_boundaries(
                file=f,
                desired_num_chunks=num_cpus,
                split_special_token=split_special_token,
            )

            chunks = []
            for i in range(len(chunk_boundaries) - 1):
                start = chunk_boundaries[i]
                end = chunk_boundaries[i + 1]
                f.seek(start)
                chunks.append((corpos_path, start, end, special_tokens))

            with Pool(processes=num_cpus) as pool:
                results = list(
                    tqdm.tqdm(
                        pool.starmap(self._process_chunk_with_boundry, chunks),
                        total=len(chunks),
                        desc="Pre-tokenizing corpus",
                    )
                )

            pre_token_count = self._merge_pre_token_counts(*results)

        end_time = time.time()
        logger.info(
            "Takes %.2f seconds to pre-tokenize the corpus file %s, speed: %.2f bytes/second",
            end_time - start_time,
            corpos_path,
            file_size / (end_time - start_time),
        )
        return pre_token_count


if __name__ == "__main__":
    file = "data/owt_valid.txt"
    split_token = b"<|endoftext|>"
    special_tokens = [b"<|endoftext|>", b"<|startoftext|>"]
    tokenizer_mp = MultiProcessPreTokenizer()
    tokenizer_mp(file, split_token, special_tokens)
    tokenizer = NativePreTokenizer()
    tokenizer(file, split_token, special_tokens)
