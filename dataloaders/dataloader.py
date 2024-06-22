import torch
import numpy as np
from torch.utils.data.distributed import DistributedSampler


class Dataloader:
    def __init__(
        self,
        config,
    ):
        self.data = config.data
        self.labels = config.labels

        self.data_dtype = config.data_dtype
        self.labels_dtype = config.labels_dtype

        self.batch_size = config.batch_size

        self.world_size = config.world_size
        self.rank = config.rank
        self.distributed = config.world_size > 1

        self.epoch_seed_mult = config.epoch_seed_mult

        self.with_labels = self.labels is not None

        self.dataset_size = len(config.data)
        self.num_batches = int(
            np.ceil(self.dataset_size / (self.batch_size * config.world_size))
        )
        self.current_batch = 0
        self.current_epoch = 0
        self.data_indices = np.arange(self.dataset_size)

        if self.distributed:
            self.distributed_sampler = DistributedSampler(
                self,
                num_replicas=self.world_size,
                rank=self.rank,
            )

    def __iter__(self):
        self.current_batch = 0

        if self.distributed:
            torch.manual_seed(self.current_epoch * self.epoch_seed_mult)
            self.distributed_sampler.set_epoch(self.current_epoch)
            self.indices = iter(self.distributed_sampler)

        else:
            np.random.seed(self.current_epoch * self.epoch_seed_mult)
            np.random.shuffle(self.data_indices)
            self.indices = iter(range(len(self)))

        self.current_epoch += 1

        return self

    def __len__(self):
        return self.num_batches

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration

        idx = next(self.indices)

        # Calculate start and end indices for the current batch
        start_idx = idx * self.batch_size
        end_idx = min(
            (start_idx + self.batch_size),
            self.dataset_size,
        )

        # Get the shuffled indices for the current batch
        batch_indices = self.data_indices[start_idx:end_idx]

        # Fetch the data using sorted indices
        batch_data = self.data[batch_indices]

        if self.with_labels:
            batch_labels = self.labels[batch_indices]

        # Convert the reordered data to a tensor
        batch = [
            torch.tensor(batch_data, dtype=self.data_dtype),
            torch.tensor(batch_labels, dtype=self.labels_dtype),
        ]

        self.current_batch += 1

        return batch
