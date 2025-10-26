import regex as re


def split_text_by_special_token(text: bytes, special_tokens: list[bytes]) -> list[bytes]:
    """
    Split the text by given special_tokens.

    Args:
        text (bytes): The text.
        special_tokens (list[bytes]): A list contains special_tokens.

    Returns:
        list[bytes]: A list that contains all splitted tokens.
    """

    escaped_tokens = [re.escape(token) for token in special_tokens]
    pattern = b"|".join(escaped_tokens)
    return re.split(pattern, text)
