from collections import defaultdict

type PreToken = bytes
type Chunk = bytes
type PreTokenCount = defaultdict[PreToken, int]
type Token = bytes
type Vocab = dict[int, Token]
type PairCount = defaultdict[tuple[Token, Token], int]
