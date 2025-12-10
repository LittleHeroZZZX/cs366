import mmap
import os
import time
from collections.abc import Iterator

from loguru import logger

from ._types import Token


def find_chunk_boundaries(
    file_path: str,
    desired_num_chunks: int,
    split_special_token: Token,
    desize_bytes: int | None = None,
) -> list[int]:
    start = time.time()
    chunk_boundaries = []
    file_size = os.path.getsize(file_path)
    if desize_bytes is not None:
        desired_num_chunks = file_size // desize_bytes

    with open(file_path, "r+b") as f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        chunk_size = file_size // desired_num_chunks

        chunk_boundaries.append(0)

        for i in range(1, desired_num_chunks):
            target_pos = i * chunk_size
            found_at = mm.find(split_special_token, target_pos)

            if found_at != -1:
                chunk_boundaries.append(found_at)
            else:
                break

        chunk_boundaries.append(file_size)
    end_time = time.time()
    logger.info(f"Chunk boundary finding took {end_time - start:.2f} seconds.")
    return sorted(list(set(chunk_boundaries)))


class FileChunkIterator:
    def __init__(self, file_path: str, boundaries: list[int], return_bytes: bool = True):
        self.file_path = file_path
        self.boundaries = boundaries
        self.return_bytes = return_bytes

    def __len__(self) -> int:
        return max(0, len(self.boundaries) - 1)

    def __iter__(self) -> Iterator[bytes] | Iterator[str]:
        with open(self.file_path, "rb") as f, mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
            for i in range(len(self.boundaries) - 1):
                start = self.boundaries[i]
                end = self.boundaries[i + 1]
                chunk = mm[start:end]

                if self.return_bytes:
                    yield chunk
                else:
                    yield chunk.decode("utf-8", errors="replace")


def make_file_str_iter(file_path: str, boundaries: list[int], return_bytes: bool = True) -> FileChunkIterator:
    return FileChunkIterator(file_path, boundaries, return_bytes)
