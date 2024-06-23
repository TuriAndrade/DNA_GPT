from dataclasses import dataclass, field


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    attn_window: int
    n_layer: int
    n_head: int
    n_embd: int
    embd_pdrop: float
    attn_pdrop: float
    resid_pdrop: float

    def gpt_large(vocab_size, block_size, attn_window=None):
        return GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            attn_window=attn_window,
            n_layer=36,
            n_head=20,
            n_embd=1280,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            resid_pdrop=0.1,
        )

    def gpt_medium(vocab_size, block_size, attn_window=None):
        return GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            attn_window=attn_window,
            n_layer=24,
            n_head=16,
            n_embd=1024,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            resid_pdrop=0.1,
        )

    def gpt_small(vocab_size, block_size, attn_window=None):
        return GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            attn_window=attn_window,
            n_layer=12,
            n_head=12,
            n_embd=768,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            resid_pdrop=0.1,
        )

    def gpt_mini(vocab_size, block_size, attn_window=None):
        return GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            attn_window=attn_window,
            n_layer=6,
            n_head=6,
            n_embd=192,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            resid_pdrop=0.1,
        )

    def gpt_micro(vocab_size, block_size, attn_window=None):
        return GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            attn_window=attn_window,
            n_layer=4,
            n_head=4,
            n_embd=128,
            embd_pdrop=0.1,
            attn_pdrop=0.1,
            resid_pdrop=0.1,
        )

    def gpt_nano(vocab_size, block_size, attn_window=None):
        return GPTConfig(
            vocab_size=vocab_size,
            block_size=block_size,
            attn_window=attn_window,
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
