import sys
import timeit
from pathlib import Path
from pprint import pprint
from typing import Annotated

import regex as re
import torch
import typer
from loguru import logger

from src.tokenization.tokenizer_trainer import TokenizerTrainer, TokenizerTrainerC

app = typer.Typer(help="Tokenizer 训练与评估工具")

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO",
)


@app.command()
def train_tokenizer(
    corpos_path: Annotated[str, typer.Argument(help="corpos file path")],
    vocab_size: Annotated[int, typer.Argument(help="target vocabulary size")],
    special_tokens: Annotated[list[str] | None, typer.Option(help="special tokens list")] = None,
    save_path: Annotated[str, typer.Option(help="save directory")] = "save",
    use_cpp: Annotated[bool, typer.Option("--cpp/--py", help="use C++ accelerated implementation")] = True,
):
    """
    Train BPE Tokenizer and save the results
    """
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]
    corpus_path_obj = Path(corpos_path)
    corpus_name = corpus_path_obj.stem
    if not corpus_path_obj.exists():
        logger.error(f"Corpus file not found: {corpos_path}")
        raise typer.Exit(code=1)

    save_dir = Path(save_path) / corpus_name
    save_dir.mkdir(parents=True, exist_ok=True)

    log_file = save_dir / "training.log"

    log_sink_id = logger.add(log_file, rotation="10 MB", retention="5 days", level="DEBUG")

    TrainerClass = TokenizerTrainerC if use_cpp else TokenizerTrainer
    backend_name = TrainerClass.__name__

    logger.info(f"Starting training for [{corpus_name}] using [{backend_name}]")
    logger.info(f"Config: vocab_size={vocab_size}, special_tokens={special_tokens}")

    try:
        tokenizer = TrainerClass(corpos_path=corpos_path, vocab_size=vocab_size, special_tokens=special_tokens)
        vocab, merges = tokenizer.train()
    except Exception:
        logger.exception("Training failed due to an error")
        raise typer.Exit(code=1) from None

    tokenizer.save(save_dir)

    longest_token_length = max(len(token) for token in tokenizer.vocab.values())
    longest_tokens = [token for token in tokenizer.vocab.values() if len(token) == longest_token_length]
    logger.success(f"Tokenizer saved to: {save_dir}")
    logger.info(f"Vocab size: {len(vocab)}")
    logger.info(f"Merges count: {len(merges)}")
    logger.info(f"Longest token length: {longest_token_length} (Tokens: {longest_tokens!r})")

    logger.remove(log_sink_id)

    return vocab, merges


@app.command()
def compare_tokenizer(
    dir1: Annotated[str, typer.Argument(help="First Tokenizer directory")],
    dir2: Annotated[str, typer.Argument(help="Second Tokenizer directory")],
    show_details: Annotated[bool, typer.Option(help="Whether to show detailed overlap parts")] = False,
):
    """
    Compare the vocabulary overlap of two trained Tokenizers
    """
    d1 = Path(dir1)
    d2 = Path(dir2)

    v1_path = d1 / "vocab.json" if (d1 / "vocab.json").exists() else d1 / "vocab.txt"
    v2_path = d2 / "vocab.json" if (d2 / "vocab.json").exists() else d2 / "vocab.txt"

    files_to_check = [v1_path, d1 / "merges.txt", v2_path, d2 / "merges.txt"]

    if not all(p.exists() for p in files_to_check):
        logger.error(f"Missing files in directories. Checked: {[str(p) for p in files_to_check]}")
        raise typer.Exit(code=1)

    def load_vocab(path: Path):
        with open(path, encoding="utf-8") as f:
            return eval(f.read())

    vocab1_dict = load_vocab(v1_path)
    vocab2_dict = load_vocab(v2_path)

    with open(d1 / "merges.txt", encoding="utf-8") as f:
        merges1_set = set(line.strip() for line in f)
    with open(d2 / "merges.txt", encoding="utf-8") as f:
        merges2_set = set(line.strip() for line in f)

    vocab1_set = set(vocab1_dict.keys())
    vocab2_set = set(vocab2_dict.keys())

    common_vocab = vocab1_set.intersection(vocab2_set)
    common_merges = merges1_set.intersection(merges2_set)

    logger.info("--- Comparison Report ---")
    logger.info(f"Dir 1: {d1} | Dir 2: {d2}")

    v_overlap = len(common_vocab) / max(len(vocab1_set), 1) * 100
    m_overlap = len(common_merges) / max(len(merges1_set), 1) * 100

    logger.info(f"Vocab overlap: {len(common_vocab)} tokens ({v_overlap:.2f}%)")
    logger.info(f"Merges overlap: {len(common_merges)} merges ({m_overlap:.2f}%)")

    if show_details:
        logger.info(f"Common tokens snippet: {list(common_vocab)[:10]}...")
        logger.info(f"Common merges snippet: {list(common_merges)[:10]}...")


@app.command()
def check_compression_ratio(
    tokenizer_path: Annotated[str, typer.Argument(help="Path to the tokenizer directory")],
    corpos_path: Annotated[str, typer.Argument(help="Path to the corpus file")],
    num_samples: Annotated[int, typer.Argument(help="Number of samples to use for compression ratio check")] = 10,
    split_special_token: str = "<|endoftext|>",
):
    """
    Check if the tokenizer compression ratio meets expectations.
    """
    from src.tokenization.tokenizer import Tokenizer

    tokenizer_dir = Path(tokenizer_path)
    vocab_file = (
        tokenizer_dir / "vocab.json" if (tokenizer_dir / "vocab.json").exists() else tokenizer_dir / "vocab.txt"
    )
    merges_file = tokenizer_dir / "merges.txt"

    if not vocab_file.exists() or not merges_file.exists():
        logger.error(f"Tokenizer files not found in: {tokenizer_path}")
        raise typer.Exit(code=1)

    tokenizer = Tokenizer.from_my_save(tokenizer_path, [split_special_token])

    total_original_size = 0
    total_encoded_size = 0
    pattern = re.compile(re.escape(split_special_token))
    # use regex to get first num_samples samples split by split_special_token
    # and calculate compression ratio
    with open(corpos_path, encoding="utf-8") as f:
        data = f.read()
        samples = pattern.split(data)[:num_samples]
        del data
    t1 = timeit.default_timer()
    all = []
    for sample in samples:
        sample_bytes = sample.encode("utf-8")
        total_original_size += len(sample_bytes)
        encoded = tokenizer.encode(sample)
        total_encoded_size += len(encoded)
        all.append((sample, encoded))
    ret = tokenizer.encode_batch(samples)
    t2 = timeit.default_timer()

    for sample, encoded in zip(samples, ret, strict=False):
        assert tokenizer.encode(sample) == encoded
        sample_bytes = sample.encode("utf-8")
        total_original_size += len(sample_bytes)
        total_encoded_size += len(encoded)
    logger.info(
        f"Encoding time for {num_samples} samples: {t2 - t1:.2f} seconds. Throughput: {total_original_size / (t2 - t1):.2f} bytes/second"
    )

    if total_original_size == 0:
        logger.error("No valid data found in the corpus for compression ratio check.")
        raise typer.Exit(code=1)

    compression_ratio = total_original_size / total_encoded_size
    logger.info(f"Compression ratio over {num_samples} samples: {compression_ratio:.2f}")


@app.command()
def encode_file(
    tokenizer_path: Annotated[str, typer.Argument(help="Path to the tokenizer directory")],
    corpos_path: Annotated[str, typer.Argument(help="Path to the corpus file")],
    output_path: Annotated[str, typer.Argument(help="Path to save the encoded output")],
    split_special_token: str = "<|endoftext|>",
):
    """
    Encode a corpus file using the specified tokenizer and save the output.
    """
    from src.tokenization.tokenizer import Tokenizer

    output = Path(output_path)
    if not output.parent.exists():
        output.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer.from_my_save(tokenizer_path, [split_special_token])

    t1 = timeit.default_timer()
    tokenizer.encode_file(corpos_path, split_token=split_special_token, save_file=output_path)
    t2 = timeit.default_timer()
    size = Path(corpos_path).stat().st_size
    logger.info(
        f"Encoding time for file {corpos_path}: {t2 - t1:.2f} seconds. Throughput: {size / (t2 - t1):.2f} bytes/second"
    )

    # out = np.array(encoded_ids, dtype=np.uint16)

    # out.tofile(output_path)

    logger.info(f"Encoded output saved to: {output}")


@app.command()
def learning_rate_tuning():
    from src.nn.optimizer import SGD

    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    lr_list = [10, 100, 1000]
    res = {}
    for lr in lr_list:
        weights_copy = weights.clone().detach().requires_grad_(True)
        opt = SGD([weights_copy], lr=lr)
        res[lr] = []
        for _ in range(10):
            opt.zero_grad()  # Reset the gradients for all learnable parameters.
            loss = (weights_copy**2).mean()  # Compute a scalar loss value.
            print(loss.cpu().item())
            loss.backward()  # Run backward pass, which computes gradients.
            opt.step()  # Run optimizer step.
            res[lr].append(loss.cpu().item())
    pprint("learning_rate_tuning on SGD:")
    pprint(res)


@app.command()
def train_model(config: str):
    from src.nn import train

    train.train(config)


@app.command()
def decode(
    model_path: Annotated[str, typer.Argument(help="Path to the trained model directory")],
    tokenizer_path: Annotated[str, typer.Argument(help="Path to the tokenizer directory")],
    prompt: Annotated[str, typer.Argument(help="Prompt text to start generation")],
    max_length: Annotated[int, typer.Argument(help="Maximum length of generated text")] = 100,
    temperature: Annotated[float, typer.Option(help="Sampling temperature")] = 1.0,
    p: Annotated[float, typer.Option(help="Nucleus sampling probability threshold")] = 0.9,
    checkpoint_number: Annotated[int | None, typer.Option(help="Checkpoint number to load")] = None,
):
    from src.nn.decode import decode

    generated_text = decode(
        prompt=prompt,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        max_length=max_length,
        temperature=temperature,
        p=p,
        checkpoint_number=checkpoint_number,
    )
    print("Generated Text:", generated_text)


if __name__ == "__main__":
    app()
