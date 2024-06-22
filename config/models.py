from dataclasses import dataclass, field


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    embd_pdrop: float
    attn_pdrop: float
    resid_pdrop: float

    def gpt_mini(vocab_size, block_size):
        return GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=6,
            n_head=6,
            n_embd=192,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            resid_pdrop=0.1,
        )

    def gpt_micro(vocab_size, block_size):
        return GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=4,
            n_head=4,
            n_embd=128,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            resid_pdrop=0.1,
        )

    def gpt_nano(vocab_size, block_size):
        return GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layer=3,
            n_head=3,
            n_embd=48,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            resid_pdrop=0.1,
        )


@dataclass
class GPTOptimConfig:
    weight_decay: float = 0.1
    learning_rate: float = 3e-4
    betas: tuple = field(default_factory=lambda: (0.9, 0.95))
    grad_norm_clip: float = 1.0
