from src.tokenization.native import perform_pretokenization

owt_valid = "data/owt_valid.txt"


def test_perform_pretokenization():
    with open(owt_valid, mode="br") as f:
        text = f.read()
    special_tokens = [rb"<|endoftext|>"]
    table = perform_pretokenization(text, special_tokens)
    print(table)


if __name__ == "__main__":
    test_perform_pretokenization()
