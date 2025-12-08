from pathlib import Path

import torch
from loguru import logger

from ..tokenization.tokenizer import Tokenizer
from .config import Config
from .functional import softmax
from .networks import TransformerLM
from .utils import load_checkpoint
import timeit


def decode(
    prompt: str,
    model_path: str,
    tokenizer_path: str,
    checkpoint_number: int | None = None,
    max_length: int = 50,
    temperature: float = 1.0,
    p: float = 0.9,
) -> str:
    """Decode a prompt into a generated text sequence using nucleus sampling.

    Args:
        prompt (str): The input text prompt to condition the generation.
        model_path (str): Path to the trained model checkpoint.
        tokenizer_path (str, optional): Path to the tokenizer files. Defaults to "".
        max_length (int, optional): Maximum length of the generated sequence. Defaults to 50.
        temperature (float, optional): Sampling temperature. Defaults to 1.0.
        p (float, optional): Nucleus sampling probability threshold. Defaults to 0.9.
    Returns:
        str: The generated text sequence.
    """
    now = timeit.default_timer()
    save_path = Path(model_path)
    assert save_path.exists(), f"Model path {model_path} does not exist."
    config_path = save_path / "config.yaml"
    assert config_path.exists(), f"Config file {config_path} does not exist."
    if checkpoint_number is None:
        checkpoint_files = list(save_path.glob("checkpoint_*.pt"))
        assert checkpoint_files, f"No checkpoint files found in {model_path}."
        checkpoint_number = max(int(f.stem.split("_")[1]) for f in checkpoint_files if f.stem.split("_")[1].isdigit())
        logger.info(f"Using checkpoint number: {checkpoint_number}")
    checkpoint_path = save_path / f"checkpoint_{checkpoint_number}.pt"
    assert checkpoint_path.exists(), f"Checkpoint file {checkpoint_path} does not exist."

    tokenizer_save_path = Path(tokenizer_path)
    assert tokenizer_save_path.exists(), f"Tokenizer path {tokenizer_path} does not exist."

    try:
        tokenizer = Tokenizer.from_my_save(tokenizer_save_path)
    except Exception as e:
        logger.error(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        raise
    config = Config.from_yaml(config_path)
    model = TransformerLM.from_config(config)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    device = torch.device(config.model.device)
    model.to(device)

    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], device=device)
    token_count = 0
    while len(input_ids) < max_length:
        token_count += 1
        logits: torch.Tensor = model(input_tensor)
        logits = logits[:, -1, :]
        logits = logits.flatten()

        logits = logits / temperature
        logits = softmax(logits, dim=-1)
        logits, index = logits.sort(dim=-1, descending=True)
        count = 0
        cum_prob = 0.0
        while cum_prob < p and count < logits.size(0):
            cum_prob += logits[count].item()
            count += 1
        filtered_logits = logits[:count]
        filtered_logits = filtered_logits / filtered_logits.sum()
        next_token_id = index[int(torch.multinomial(filtered_logits, num_samples=1).item())].item()
        generated_ids = input_ids + [next_token_id]
        input_tensor = torch.tensor([generated_ids], device=device)
        input_ids = generated_ids

        next_token = tokenizer.decode([next_token_id])
        if next_token == "<|endoftext|>":
            break

    generated_text = tokenizer.decode(input_ids)
    logger.info(f"Generate speed: {token_count / (timeit.default_timer() - now):.2f} tok/s")
    return generated_text
