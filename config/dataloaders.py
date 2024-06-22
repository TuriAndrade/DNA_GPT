from dataclasses import dataclass, field
from typing import Any
import numpy as np
import torch


@dataclass
class DataloaderConfig:
    data: np.ndarray
    labels: np.ndarray
    data_dtype: Any
    labels_dtype: Any
    batch_size: int
    world_size: int
    rank: int
    epoch_seed_mult: int


@dataclass
class LoadSeqDataConfig:
    file_path: str
    skip_first_line: bool
    seq_len: int
    val_size: float
    test_size: float
    seed: int
    dataset_name: str = "DNA dataset"
    seq_vocab: list = field(
        default_factory=lambda: ["A", "C", "G", "T", "N"],
    )
    dataloader_config: dict = field(
        default_factory=lambda: {
            "data_dtype": torch.long,
            "labels_dtype": torch.long,
            "batch_size": 2,
            "epoch_seed_mult": 1,
        },
    )
