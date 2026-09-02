# model.py

import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.init as init

# -----------------------
# Causal Self-Attention
# -----------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        return self.weight * x

default_init = "default"

def make_linear(in_dim, out_dim, bias=True, init_type="zero"):
    layer = nn.Linear(in_dim, out_dim, bias=bias)
    if init_type == "default":
        pass

    elif init_type == "zero":
        init.zeros_(layer.weight)
        if layer.bias is not None:
            init.zeros_(layer.bias)

    elif init_type == "normal":
        sigma = 1 / in_dim**0.5
        init.normal_(layer.weight, mean=0.0, std=sigma)
        if layer.bias is not None:
            init.normal_(layer.bias, mean=0.0, std=sigma)
    else:
        raise ValueError(f"Unknown init_type: {init_type}")

    return layer

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = make_linear(embed_dim, embed_dim, init_type=default_init)
        self.key   = make_linear(embed_dim, embed_dim, init_type=default_init)
        self.value = make_linear(embed_dim, embed_dim, init_type=default_init)
        self.out   = make_linear(embed_dim, embed_dim, init_type="zero")
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)





class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout, act="gelu"):
        super().__init__()
        self.act = act
        self.fc1 = make_linear(embed_dim, hidden_dim, init_type=default_init)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = make_linear(hidden_dim, embed_dim, init_type="zero")

    def custom_act(self, x):
        if self.act == "relu":
            return F.relu(x)
        if self.act == "gelu":
            return F.gelu(x)
        if self.act == "erf":
            return torch.erf(x)

        raise ValueError(f"Unknown activation: {self.act}")
    def forward(self, x):
        a = self.fc1(x)
        a = self.custom_act(a)
        a = self.drop1(a)
        a = self.fc2(a)
        return a

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, dropout, activation):
        super().__init__()
        self.ln1 = RMSNorm(embed_dim)
        self.ln2 = RMSNorm(embed_dim)

        self.attn = CausalSelfAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, ff_hidden_dim, dropout, activation)

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))

        return x


# -----------------------
# Causal Mask
# -----------------------
def causal_mask(T, device):
    return torch.tril(torch.ones(T, T, device=device)).unsqueeze(0).unsqueeze(0)

import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super().__init__()

        pe = torch.zeros(seq_len, embed_dim)

        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # shape: (1, seq_len, embed_dim)
        pe = pe.unsqueeze(0)

        # register_buffer means:
        # - not trainable
        # - saved in state_dict
        # - moved automatically to cuda/cpu with model.to(device)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: (B, T, embed_dim)
        """
        T = x.size(1)
        return x + self.pe[:, :T, :]

# -----------------------
# Decoder-Only Transformer LM
# -----------------------
# import math
# import torch
# import torch.nn as nn

from models_FFN import *

class TF_classical(nn.Module):
    def __init__(
        self,
        vocab_size,
        seq_len,
        embed_dim=128,
        depth=4,
        num_heads=8,
        ff_hidden_dim=128,
        dropout=0.1,
        activation="gelu",
    ):
        super().__init__()

        self.seq_len = seq_len

        self.token_table = nn.Embedding(vocab_size, embed_dim)

        # Sinusoidal positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(seq_len, embed_dim)

        # Optional dropout after token + position embedding
        # self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim,
                num_heads,
                ff_hidden_dim,
                dropout,
                activation,
            )
            for _ in range(depth)
        ])

        self.lm_head = Proj_block(embed_dim, vocab_size, composed=True)

    def forward(self, x):
        B, T = x.shape
        assert T <= self.seq_len

        x = self.token_table(x)          # (B, T, embed_dim)
        x = self.pos_encoding(x)            # (B, T, embed_dim)

        mask = causal_mask(T, x.device)

        for block in self.blocks:
            x = block(x, mask)

        return self.lm_head(x)




class TF_same_block(nn.Module):
    def __init__(
        self,
        vocab_size,
        seq_len,
        embed_dim=128,
        depth=4,
        num_heads=8,
        ff_hidden_dim=128,
        dropout=0.1,
        activation="gelu",
    ):
        super().__init__()

        self.seq_len = seq_len

        self.token_table = nn.Embedding(vocab_size, embed_dim)

        # Sinusoidal positional encoding
        self.pos_encoding = SinusoidalPositionalEncoding(seq_len, embed_dim)

        # Optional dropout after token + position embedding
        # self.dropout = nn.Dropout(dropout)
        self.depth = depth
        self.block = TransformerBlock(
                embed_dim,
                num_heads,
                ff_hidden_dim,
                dropout,
                activation,)


        self.lm_head = Proj_block(embed_dim, vocab_size, composed=True)

    def forward(self, x):
        B, T = x.shape
        assert T <= self.seq_len

        x = self.token_table(x)          # (B, T, embed_dim)
        x = self.pos_encoding(x)            # (B, T, embed_dim)

        mask = causal_mask(T, x.device)

        for _ in range(self.depth):
            x = self.block(x, mask)

        return self.lm_head(x)