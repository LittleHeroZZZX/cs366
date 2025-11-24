import heapq
import timeit
from collections import defaultdict
from pathlib import Path

import tokenizer_cpp
import tqdm
from line_profiler import profile
from loguru import logger

from ._types import Token, Vocab
from .pre_tokenization import MultiProcessPreTokenizer


class ComparablePair:
    __slots__ = ("pair",)

    def __init__(self, pair):
        self.pair = pair

    def __lt__(self, other):
        return self.pair > other.pair

    def __eq__(self, other):
        return self.pair == other.pair

    def __repr__(self):
        return str(self.pair)


class TokenizerTrainerBase:
    vocab: Vocab
    merges: list[tuple[bytes, bytes]]

    def train(self) -> tuple[Vocab, list[tuple[bytes, bytes]]]:
        raise NotImplementedError

    def save(self, directory: str | Path) -> None:
        import json

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        vocab_file = directory / "vocab.json"
        merges_file = directory / "merges.txt"

        vocab_to_save = {}

        for token_id, token_bytes in self.vocab.items():
            token_str = token_bytes.decode("latin-1")

            vocab_to_save[token_str] = token_id
        vocab_to_save = {k: v for k, v in sorted(vocab_to_save.items(), key=lambda item: item[1])}

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab_to_save, f, indent=2, ensure_ascii=True)

        with open(merges_file, "w", encoding="utf-8") as f:
            for p1, p2 in self.merges:
                s1 = p1.decode("latin-1")

                s2 = p2.decode("latin-1")

                f.write(f"{s1} {s2}\n")

        logger.info(f"Model saved to {directory}")


class TokenizerTrainer(TokenizerTrainerBase):
    def __init__(
        self,
        corpos_path: str,
        vocab_size: int,
        special_tokens: list[str],
        split_special_token: Token = b"<|endoftext|>",
        pre_tokenizer_cls=MultiProcessPreTokenizer,  # type: ignore
    ) -> None:
        self.pre_tokenizer = pre_tokenizer_cls()
        self.target_vocab_size = vocab_size
        self.special_tokens = [token.encode() for token in special_tokens]
        self.corpos_path = corpos_path
        self.split_special_token = split_special_token

        self.vocab_size = 0
        self.vocab = {}  # type: ignore
        self.pair_counts = defaultdict(int)  # type: ignore
        self.merges: list[tuple[bytes, bytes]] = []
        self.pre_token_states: dict[bytes, list[bytes]] = {}

        self.pair_to_pretokens: dict[tuple[bytes, bytes], set[bytes]] = defaultdict(set)

        self.pair_heap = []

    def _init(self) -> None:
        for token in self.special_tokens:
            self._add_token(token)
        for ascii_code in range(256):
            self._add_token(bytes([ascii_code]))

        self.pre_token_count = self.pre_tokenizer(
            corpos_path=self.corpos_path,
            split_special_token=self.split_special_token,
            special_tokens=self.special_tokens,
        )

        for pre_token, count in self.pre_token_count.items():
            self.pre_token_states[pre_token] = [bytes([b]) for b in pre_token]
            for idx in range(len(self.pre_token_states[pre_token]) - 1):
                pair = (self.pre_token_states[pre_token][idx], self.pre_token_states[pre_token][idx + 1])
                self.pair_counts[pair] += count
                self.pair_to_pretokens[pair].add(pre_token)

        for pair, count in self.pair_counts.items():
            heapq.heappush(self.pair_heap, (-count, ComparablePair(pair)))

    def _determine_merge_pair(self) -> tuple[Token, Token] | None:
        while self.pair_heap:
            neg_count, wrapper = heapq.heappop(self.pair_heap)
            count = -neg_count
            pair = wrapper.pair

            if count == self.pair_counts[pair]:
                return pair

        return None

    def _merge_pair(self, pair: tuple[Token, Token]) -> None:
        merged_token = pair[0] + pair[1]
        self._add_token(merged_token)

        affected_pre_tokens = self.pair_to_pretokens[pair]

        for pre_token in affected_pre_tokens:
            state = self.pre_token_states[pre_token]
            count = self.pre_token_count[pre_token]

            idx = 0
            while idx < len(state) - 1:
                if (state[idx], state[idx + 1]) == pair:
                    state[idx] = merged_token
                    state.pop(idx + 1)

                    if idx > 0:
                        last_token = state[idx - 1]
                        old_prev_pair = (last_token, pair[0])
                        self.pair_counts[old_prev_pair] -= count
                        heapq.heappush(
                            self.pair_heap, (-self.pair_counts[old_prev_pair], ComparablePair(old_prev_pair))
                        )

                        new_prev_pair = (last_token, merged_token)
                        self.pair_counts[new_prev_pair] += count
                        self.pair_to_pretokens[new_prev_pair].add(pre_token)
                        heapq.heappush(
                            self.pair_heap, (-self.pair_counts[new_prev_pair], ComparablePair(new_prev_pair))
                        )

                    if idx < len(state) - 1:
                        next_token = state[idx + 1]
                        old_next_pair = (pair[1], next_token)
                        self.pair_counts[old_next_pair] -= count
                        heapq.heappush(
                            self.pair_heap, (-self.pair_counts[old_next_pair], ComparablePair(old_next_pair))
                        )

                        new_next_pair = (merged_token, next_token)
                        self.pair_counts[new_next_pair] += count
                        self.pair_to_pretokens[new_next_pair].add(pre_token)
                        heapq.heappush(
                            self.pair_heap, (-self.pair_counts[new_next_pair], ComparablePair(new_next_pair))
                        )

                else:
                    idx += 1

        del self.pair_counts[pair]
        del self.pair_to_pretokens[pair]

    def _add_token(self, token: Token) -> None:
        self.vocab[self.vocab_size] = token
        self.vocab_size += 1

    def train(self) -> tuple[Vocab, list[tuple[bytes, bytes]]]:
        self._init()

        tqdm.tqdm.write(f"Initial vocabulary size: {self.vocab_size}")
        num_merges_needed = self.target_vocab_size - self.vocab_size
        if num_merges_needed <= 0:
            return {}, self.merges

        with tqdm.tqdm(total=num_merges_needed, desc="Training BPE") as pbar:
            while self.vocab_size < self.target_vocab_size:
                merge_pair = self._determine_merge_pair()
                if merge_pair is None:
                    break

                self.merges.append(merge_pair)
                self._merge_pair(merge_pair)

                pbar.update(1)
                pbar.set_description(f"Vocab size: {self.vocab_size}")

        tqdm.tqdm.write(f"Training complete. Final vocab size: {self.vocab_size}")
        return self.vocab, self.merges


@profile
class TokenizerTrainerC(TokenizerTrainerBase):
    def __init__(
        self,
        corpos_path: str,
        vocab_size: int,
        special_tokens: list[str],
        split_special_token: Token = b"<|endoftext|>",
        pre_tokenizer_cls=MultiProcessPreTokenizer,
    ):
        self.pre_tokenizer = pre_tokenizer_cls()
        self.target_vocab_size = vocab_size
        self.special_tokens = [token.encode() for token in special_tokens]
        self.corpos_path = corpos_path
        self.split_special_token = split_special_token

        self.vocab_size = 0
        self.vocab = {}  # type: ignore
        self.pair_counts = defaultdict(int)  # type: ignore
        self.merges: list[tuple[bytes, bytes]] = []
        self.pre_token_states: dict[bytes, list[bytes]] = {}

        self.trainer = tokenizer_cpp.TokenizerTrainerC(vocab_size)

    def train(self) -> tuple[dict, list[tuple[bytes, bytes]]]:
        self.pre_token_count = self.pre_tokenizer(
            corpos_path=self.corpos_path,
            split_special_token=self.split_special_token,
            special_tokens=self.special_tokens,
        )
        self.trainer.LoadData(self.special_tokens, self.pre_token_count)
        t1 = timeit.default_timer()
        self.vocab, self.merges = self.trainer.train()
        t1 = timeit.default_timer() - t1
        logger.info(
            f"TokenizerTrainerC training time: {t1:.2f} seconds, {self.target_vocab_size / t1:.2f} tokens/second"
        )
        return self.vocab, self.merges
