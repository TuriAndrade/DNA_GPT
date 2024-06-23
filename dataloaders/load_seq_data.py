from .dataloader import Dataloader
from config.dataloaders import DataloaderConfig
import numpy as np
import torch


class LoadSeqData:
    def __init__(
        self,
        config,
    ):
        self.file_path = config.file_path
        self.skip_first_line = config.skip_first_line
        self.seq_vocab = config.seq_vocab
        self.seq_len = config.seq_len
        self.val_size = config.val_size
        self.test_size = config.test_size
        self.seed = config.seed
        self.dataset_name = config.dataset_name
        self.dataloader_config = config.dataloader_config

        self.sos_token = 0
        self.seq_encoding_map = {
            token: (idx + 1) for idx, token in enumerate(self.seq_vocab)
        }

        self.str_sequences = []
        self.sequences = []
        self.labels = []

        self.train_sequences = []
        self.train_labels = []

        self.val_sequences = []
        self.val_labels = []

        self.test_sequences = []
        self.test_labels = []

        self.build()

    def read_input(self):
        first_line = True

        with open(self.file_path, "r") as file:
            for line in file:
                if first_line and self.skip_first_line:
                    first_line = False
                    continue
                if line.strip():
                    parts = line.strip().split("\t")
                    self.str_sequences.append(parts[0])
                    self.labels.append(int(parts[1]))

    def encode_sequences(self):
        self.sequences = [
            [self.seq_encoding_map[nucleotide] for nucleotide in seq]
            for seq in self.str_sequences
        ]

    def split_into_subsequences_helper(self, sequence):
        subsequences = []

        for i in range(0, len(sequence), self.seq_len):
            subseq = sequence[i : i + self.seq_len]
            if len(subseq) == self.seq_len:
                subsequences.append(subseq)

        return subsequences

    def split_into_subsequences(self):
        all_subsequences = []
        all_labels = []

        for seq, label in zip(self.sequences, self.labels):
            subsequences = self.split_into_subsequences_helper(seq)
            all_subsequences.extend(subsequences)
            all_labels.extend([label] * len(subsequences))

        self.sequences = all_subsequences
        self.labels = all_labels

    def to_numpy(self):
        self.sequences = np.array(self.sequences)
        self.labels = np.array(self.labels)

    def train_test_split(self):
        np.random.seed(self.seed)

        indices = np.arange(self.sequences.shape[0])
        np.random.shuffle(indices)

        self.sequences = self.sequences[indices]
        self.labels = self.labels[indices]

        num_train = int(self.sequences.shape[0] * (1 - self.val_size - self.test_size))
        num_val = int(self.sequences.shape[0] * self.val_size)

        self.train_sequences = self.sequences[:num_train]
        self.train_labels = self.labels[:num_train]

        self.val_sequences = self.sequences[num_train : num_train + num_val]
        self.val_labels = self.labels[num_train : num_train + num_val]

        self.test_sequences = self.sequences[num_train + num_val :]
        self.test_labels = self.labels[num_train + num_val :]

    def build(self):
        self.read_input()
        self.encode_sequences()
        self.split_into_subsequences()
        self.to_numpy()
        self.train_test_split()

    def get_vocab_size(self, with_sos=True):
        if with_sos:
            return len(self.seq_vocab) + 1

        else:
            return len(self.seq_vocab)

    def get_train_sequences_prior(self, with_sos=True):
        counts = np.unique(self.train_sequences, return_counts=True)[1]

        if with_sos:
            sos_count = self.train_sequences.shape[0]
            counts = np.insert(counts, 0, sos_count)

        else:
            counts = np.insert(counts, 0, 0)

        return counts / counts.sum()

    def get_train_loader(self, world_size=1, rank=0):
        config = DataloaderConfig(
            data=self.train_sequences,
            labels=self.train_labels,
            world_size=world_size,
            rank=rank,
            **self.dataloader_config,
        )

        return Dataloader(
            config,
        )

    def get_val_loader(self, world_size=1, rank=0):
        config = DataloaderConfig(
            data=self.val_sequences,
            labels=self.val_labels,
            world_size=world_size,
            rank=rank,
            **self.dataloader_config,
        )

        return Dataloader(
            config,
        )

    def get_test_loader(self, world_size=1, rank=0):
        config = DataloaderConfig(
            data=self.test_sequences,
            labels=self.test_labels,
            world_size=world_size,
            rank=rank,
            **self.dataloader_config,
        )

        return Dataloader(
            config,
        )

    def prepend_sos_token(self, tensor, crop_end=True):
        B, T = tensor.shape

        sos_token = torch.zeros(B, 1) + self.sos_token

        out = torch.cat((sos_token, tensor), dim=1).type(tensor.dtype)

        if crop_end:
            return out[:, :-1]

        else:
            return out
