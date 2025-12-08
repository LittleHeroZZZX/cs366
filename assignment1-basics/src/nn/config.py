from pathlib import Path
from typing import Self

import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    vocab_size: int
    context_length: int
    num_layers: int
    hidden_dim: int
    inner_dim: int
    num_heads: int
    theta: float
    device: str = "cpu"
    dtype: str = "float32"


class OptimizerConfig(BaseModel):
    learning_rate: float
    weight_decay: float
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


class TrainingConfig(BaseModel):
    batch_size: int
    total_iterations: int
    warmup_iterations: int
    cosine_cycle_iterations: int
    max_l2_norm: float
    checkpoint_interval: int
    output_dir: str
    train_data: str
    val_data: str
    save_step: int = 500
    wandb_project: str = "transformer-lm"
    wandb_run_name: str | None = None
    log_step: int = 50

    @classmethod
    def from_yaml(cls, path: str) -> Self:
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


class Config(BaseModel):
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(), f)
