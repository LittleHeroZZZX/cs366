from collections import defaultdict
from .pre_tokenization import PreTokenizer, MultiProcessPreTokenizer
from .types import PairCount, Vocab
import tqdm


class Tokenizer:
    def __init__(
        self,
        corpos_path: str,
        vocab_size: int,
        special_tokens: list[str],
        split_special_token: bytes = b"<|endoftext|>",
        pre_tokenizer_cls: type[PreTokenizer] = MultiProcessPreTokenizer,
    ) -> None:
        self.pre_tokenizer: PreTokenizer = pre_tokenizer_cls()
        self.target_vocab_size = vocab_size
        self.special_tokens = [token.encode() for token in special_tokens]
        self.corpos_path = corpos_path
        self.split_special_token = split_special_token

        self.vocab_size = 0
        self.vocab: Vocab = {}
        self.pair_counts: PairCount = defaultdict(int)
        self.merges: list[tuple[bytes, bytes]] = []
        self.pre_token_states: dict[bytes, list[bytes]] = {}  # current state of each word during BPE training

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

    def _determine_merge_pair(self) -> tuple[bytes, bytes] | None:
        if not self.pair_counts:
            return None

        most_frequent_pair, count = None, -1
        for pair, pair_count in self.pair_counts.items():
            if pair_count > count:
                most_frequent_pair = pair
                count = pair_count
            elif pair_count == count and most_frequent_pair is not None:
                if pair > most_frequent_pair:
                    most_frequent_pair = pair

        return most_frequent_pair

    def _merge_pair(self, pair: tuple[bytes, bytes]) -> None:
        merged_token = pair[0] + pair[1]
        self._add_token(merged_token)

        new_pair_counts: PairCount = defaultdict(int)

        for pre_token, state in self.pre_token_states.items():
            new_state = []
            idx = 0
            while idx < len(state):
                if idx < len(state) - 1 and (state[idx], state[idx + 1]) == pair:
                    new_state.append(merged_token)
                    idx += 2
                else:
                    new_state.append(state[idx])
                    idx += 1
            self.pre_token_states[pre_token] = new_state

            for i in range(len(new_state) - 1):
                new_pair = (new_state[i], new_state[i + 1])
                new_pair_counts[new_pair] += self.pre_token_count[pre_token]

        self.pair_counts = new_pair_counts

    def _add_token(self, token: bytes) -> None:
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


if __name__ == "__main__":
    path = "data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>", "<|pad|>"]
    tokenizer = Tokenizer(corpos_path=path, vocab_size=vocab_size, special_tokens=special_tokens)
    print(tokenizer.train())
