import os
from dataclasses import dataclass
from .models import GPTConfig, GPTOptimConfig
from .dataloaders import LoadSeqDataConfig


@dataclass
class DNAGPTTrainerConfig:
    model_config: GPTConfig
    optim_config: GPTOptimConfig
    load_seq_data_config: LoadSeqDataConfig
    epochs: int
    batch_size: int
    save_path: str = None
    master_addr: str = "localhost"
    master_port: str = "1234"
    backend: str = "nccl"
    main_device: int = 0
    process_timeout: int = 10000
