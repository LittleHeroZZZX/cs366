import re
import time
from pathlib import Path

import loguru
import numpy as np
import torch

import wandb

from . import functional as F
from .config import Config
from .networks import TransformerLM
from .optimizer import AdamW
from .utils import get_lr_cosine_schedule, gradient_clipping, load_batch, load_checkpoint, save_checkpoint

logger = loguru.logger


def train(config: Config | str):
    """
    Train the model with specified config.

    Ths function support resume from checkpoint.

    Args:
        config (Config | str): The training configuration or path to the yaml file.
    """
    if isinstance(config, str):
        config = Config.from_yaml(config)

    assert isinstance(config, Config)

    output_dir = Path(config.training.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        config.to_yaml(output_dir / "config.yaml")

    wandb.init(
        project=config.training.wandb_project,
        name=config.training.wandb_run_name,
        config=config.model_dump(),
    )

    model = TransformerLM.from_config(config)
    optimizer = AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        betas=(config.optimizer.beta1, config.optimizer.beta2),
        eps=config.optimizer.eps,
    )

    # load model form max checkpoint if exists
    checkpoint_regex = re.compile(r"checkpoint_(\d+)\.pt")
    max_iteration = -1
    for file in output_dir.iterdir():
        match = checkpoint_regex.match(file.name)
        if match:
            iteration = int(match.group(1))
            if iteration > max_iteration:
                max_iteration = iteration
    current_iteration = 0

    if max_iteration >= 0:
        logger.info(f"Resuming from checkpoint at iteration {max_iteration}")
        checkpoint_path = output_dir / f"checkpoint_{max_iteration}.pt"
        current_iteration = load_checkpoint(checkpoint_path, model, optimizer)
        current_iteration += 1
    else:
        model._reset_params()

    assert current_iteration < config.training.total_iterations

    logger.info(f"Loading training dataset from {config.training.train_data}")
    train_dataset = np.memmap(config.training.train_data, dtype=np.uint16)

    logger.info(f"Starting training from iteration {current_iteration} to {config.training.total_iterations - 1}")
    # Training loop
    model = model.to(torch.device(config.model.device))
    model.train()
    total_tokens = 0
    t1 = time.time()
    for iteration in range(current_iteration + 1, config.training.total_iterations + 1):
        optimizer.zero_grad()
        lr = get_lr_cosine_schedule(
            iteration,
            config.optimizer.learning_rate,
            0.0,
            config.training.warmup_iterations,
            config.training.cosine_cycle_iterations,
        )
        for param_group in optimizer.param_groups:
            param_group["alpha"] = lr
        x_batch, y_batch = load_batch(
            train_dataset,
            config.training.batch_size,
            config.model.context_length,
            torch.device(config.model.device),
        )
        logits = model(x_batch)
        loss = F.cross_entropy_loss(logits, y_batch)
        loss.backward()
        gradient_clipping(model.parameters(), config.training.max_l2_norm)
        optimizer.step()

        if iteration % config.training.log_step == 0:
            t2 = time.time()
            elapsed = t2 - t1
            batch_per_sec = config.training.batch_size / elapsed
            tokens_per_sec = batch_per_sec * config.model.context_length
            total_tokens += config.training.batch_size * config.model.context_length
            t1 = time.time()
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/iteration": iteration,
                    "train/lr": lr,
                    "train/batch_per_sec": batch_per_sec,
                    "train/tokens_per_sec": tokens_per_sec,
                    "train/total_tokens": total_tokens,
                }
            )
            logger.info(f"Iteration {iteration}: train loss = {loss.item():.4f}")

        if iteration != 0 and iteration % config.training.save_step == 0:
            save_path = output_dir / f"checkpoint_{iteration}.pt"
            save_checkpoint(model, optimizer, iteration, save_path)
    wandb.finish()
