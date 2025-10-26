from collections import defaultdict

vocab_T = dict[bytes, int]
merges_T = list[tuple[bytes, bytes]]
word_count_T = defaultdict[bytes, int]
