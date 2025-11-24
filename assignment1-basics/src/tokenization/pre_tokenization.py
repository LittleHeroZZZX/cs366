import os
import time
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from collections.abc import Iterator
from multiprocessing import Pool, cpu_count

import regex as re
import tqdm
from line_profiler import profile
from loguru import logger

from ._types import Chunk, PreTokenCount, Token
from .utils import find_chunk_boundaries


@profile
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
        pattern = re.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        for mini_chunk in re.split(b"|".join([re.escape(token) for token in special_tokens]), chunk):
            pre_tokens = Counter(re.findall(pattern, mini_chunk))
            for pre_token, count in pre_tokens.items():
                pre_token_count[pre_token] += count
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
                file_path=corpos_path,
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
            "Takes {:.2f} seconds to pre-tokenize the corpus file {:.2f}, speed: {:.2f} bytes/second",
            end_time - start_time,
            corpos_path,
            file_size / (end_time - start_time),
        )
        return pre_token_count


@profile
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
        final_pre_token_count: PreTokenCount = defaultdict(int)

        start_time = time.time()
        file_size = os.path.getsize(corpos_path)
        num_cpus = cpu_count()

        desired_chunks = num_cpus * 100

        chunk_boundaries = find_chunk_boundaries(
            file_path=corpos_path,
            desired_num_chunks=desired_chunks,
            split_special_token=split_special_token,
        )

        chunks_args = []
        for i in range(len(chunk_boundaries) - 1):
            start = chunk_boundaries[i]
            end = chunk_boundaries[i + 1]
            chunks_args.append((corpos_path, start, end, special_tokens))

        logger.info(f"Splitting task into {len(chunks_args)} chunks.")

        with Pool(processes=num_cpus * 2) as pool:
            chunk_iter = pool.imap_unordered(self._worker_wrapper, chunks_args)

            for chunk_result in tqdm.tqdm(chunk_iter, total=len(chunks_args), desc="Pre-tokenizing"):
                for token, count in chunk_result.items():
                    final_pre_token_count[token] += count

        end_time = time.time()
        logger.info(
            "Takes {:.2f} seconds to pre-tokenize, speed: {:.2f} bytes/second",
            end_time - start_time,
            file_size / (end_time - start_time),
        )
        return final_pre_token_count

    @staticmethod
    def _worker_wrapper(args):
        tokenizer_instance = MultiProcessPreTokenizer()
        return tokenizer_instance._process_chunk_with_boundry(*args)


if __name__ == "__main__":
    file = "data/owt_valid.txt"
    split_token = b"<|endoftext|>"
    special_tokens = [b"<|endoftext|>", b"<|startoftext|>"]
    tokenizer_mp = MultiProcessPreTokenizer()
    tokenizer_mp(file, split_token, special_tokens)
    tokenizer = NativePreTokenizer()
    tokenizer(file, split_token, special_tokens)
