import math
from dataclasses import dataclass
import torch
import torch.nn as nn

from torch.nn import functional as F

@dataclass
class GPTConfig:
    block_size: int = 256  # context window size
    vocab_size: int = 65  # max number of distinct tokens
    n_layer: int = 6  # N = number of blocks
    n_head: int = 6
    n_embd: int = 384


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query and value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)  # Wo

        self.n_embd = config.n_embd
        self.n_head = config.n_head

        # here bias = mask, following the OpenAI/HF naming
        self.register_buffer("bias",
                             torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size,
                                                                                               config.block_size)) # 1,1,T,T

    def forward(self, x):
        B, T, C = x.size()  # batch_size, sequence_length, embedding_dimensionality
        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # nh = number of heads, hs = head size (C/nh) C = nh * hs
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # B, nh, T, hs
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = ((q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))))  # B, nh, T, T
        # masking
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # B, nh, T, T
        att = F.softmax(att, dim=-1)
        y = att @ v # (B, nh, T, T) @ (B, nh, T, hs) => (B, nh, T, hs)
        y = y.transpose(1,2).contiguous().view(B,T,C) # cat all heads side by side

        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),  # embedding table for tokens
            wpe=nn.Embedding(config.block_size, config.n_embd),  # embedding table for positons
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # Nx
            ln_f=nn.LayerNorm(config.n_embd)  # layer normalization
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # Introduced in GPT-2
