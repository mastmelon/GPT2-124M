import math
from dataclasses import dataclass
import torch
import torch.nn as nn

from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024  # max sequence length
    vocab_size: int = 50257  # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12  # number of layers
    n_head: int = 12  # number of heads
    n_embd: int = 768  # embedding dimension


class CausalSelfAttention(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # key, query and value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)  # Wo
        self.c_proj.NANOGPT_SCALE_INIT = 1  # attaching this flag

        self.n_embd = config.n_embd
        self.n_head = config.n_head

        # here bias = mask, following the OpenAI/HF naming
        self.register_buffer("bias",
                             torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size,
                                                                                               config.block_size))  # 1,1,T,T

    def forward(self, x):
        B, T, C = x.size()  # batch_size, sequence_length, embedding_dimensionality
        qkv = self.c_attn(x)  # (B, T, 3 * C)
        q, k, v = qkv.split(self.n_embd, dim=2)
        # nh = number of heads, hs = head size (C/nh) C = nh * hs
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # B, nh, T, hs
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # att = ((q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))))  # B, nh, T, T
        # # masking
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))  # B, nh, T, T
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B, nh, T, T) @ (B, nh, T, hs) => (B, nh, T, hs)

        # Flash attention
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)  # cat all heads side by side

        # output projection
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1  # attaching this flag

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

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight  # copying the data pointer or reference
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5  # 2 because every block has 2 additions
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        # forward the token and position embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)  # shape (T)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # flatten (B,T) -> single dimension | targets also flatten
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2': dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            'gpt2-large': dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            'gpt2-xl': dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]  # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


# ----------------------------------------------- # -----------------------------------------------

# ======== attempt to autodetect device ==========

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# ========  ========== ========== ========== ======

import time
import tiktoken


class DataLoaderLite:

    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('input.txt', 'r') as f:
            text = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(text)

        self.tokens = torch.tensor(tokens)
        print(f"loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current_position = 0

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position: self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets

        self.current_position += B * T

        if self.current_position + (B * T + 1) > len(self.tokens):
            self.current_position = 0

        return x, y


train_loader = DataLoaderLite(B=4, T=32) # by default we want to max out the Batch size which fits on GPU
torch.set_float32_matmul_precision('high') # using Tensorflow32, TF32. Tell pytorch what kind of kernels to run
# ^ Expect only computations to be faster by factor of 8. For storing nums we are still using float32 and memory movement
# limits still apply


# get logits
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# model = torch.compile(model)

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50

def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)


optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
for i in range(50):
    t0 = time.time()
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)  # move tensors to GPU
    optimizer.zero_grad()

    # https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # determine and set the learning rate for this iteration
    lr = get_lr(i)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    optimizer.step()
    # torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1 - t0) * 1000  # milisecs
    tokens_per_sec = (train_loader.B * train_loader.T) / (t1 - t0)
    print(f"step {i}, loss: {loss.item()}, norm: {norm:.4f} dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}")

import sys;

sys.exit(0)

num_return_sequence = 5
max_len = 30

# # input tokens
# import tiktoken
# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a language model,")
# tokens = torch.tensor(tokens, dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequence, 1) # (5, #tokens)
# x = tokens.to(device)

torch.manual_seed(42)
# torch.cuda.manual_seed(42)

while x.size(1) < max_len:
    with torch.no_grad():
        logits = model(x)  # (B, T, vocab_size)
        # Take the logits of last token
        logits = logits[:, -1, :]  # B,vocab_size
        # Prob using softmax
        probs = F.softmax(logits, -1)

        # top-k prob
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1)  # (B,1)
        # gather corresponding indices = token numerical value
        xcol = torch.gather(topk_indices, -1, ix)  # (B,1)

        x = torch.cat((x, xcol), dim=1)  # (B, T) (B,1)

# print the generated text

for i in range(num_return_sequence):
    tokens = x[i, :max_len].tolist()
    decoded_tokens = enc.decode(tokens)
    print(">", decoded_tokens)
