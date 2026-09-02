
# model.py
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# -----------------------
# Causal Self-Attention
# -----------------------

import torch
import torch.nn as nn
import torch.nn.init as init


default_init = "default"

def make_linear(in_dim, out_dim, bias=True, init_type="zero"):
    layer = nn.Linear(in_dim, out_dim, bias=bias)
    if init_type == "default":
        pass

    elif init_type == "zero":
        init.zeros_(layer.weight)
        if layer.bias is not None:
            init.zeros_(layer.bias)

    return layer


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: (..., dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        # return self.weight * x
        return x

def freeze(linear):
    linear.weight.requires_grad = False

    if linear.bias is not None:
        linear.bias.requires_grad = False

def custom_act(x, act):
    if act == "relu":
        return F.relu(x)
    if act == "gelu":
        return F.gelu(x)
    if act == "id":
        return x
    raise ValueError(f"Unknown activation: {act}")

class Proj_block(nn.Module):
    def __init__(self, input_dim, output_dim, init=default_init, composed = True):
        super().__init__()
        self.act = "id"
        hidden_dim = min(input_dim, output_dim)
        self.hidden_dim = hidden_dim
        self.composed = composed
        self.proj_base = make_linear(input_dim, output_dim, init_type=init)
        if self.composed:
            freeze(self.proj_base)
            self.sensor = make_linear(input_dim, hidden_dim, init_type=init)
            self.response = make_linear(hidden_dim, output_dim, init_type="zero")
    def forward(self, x):
        if self.composed:
            a = self.sensor(x)
            a = custom_act(a, self.act)
            a = self.response(a)
            return self.proj_base(x) + a
        else:
            a = self.proj_base(x)
            return a

class FFN_block(nn.Module):
    def __init__(self, embed_dim, hidden_dim, act, init=default_init):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.act = act
        self.sensor = make_linear(embed_dim, hidden_dim, bias=True, init_type=init)
        self.response = make_linear(hidden_dim, embed_dim, bias=False, init_type="zero")
    def forward(self, x):
        a = self.sensor(x)
        a = custom_act(a, self.act)
        a = self.response(a)
        return a

class ResNet_std(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, depth, output_dim, act, composed_proj = True ):
        super().__init__()
        self.embed_dim = embed_dim
        self.depth = depth
        self.fc_in = Proj_block(input_dim, embed_dim, init=default_init, composed=composed_proj)
        self.fc_out = Proj_block(embed_dim, output_dim, init=default_init, composed=composed_proj)
        self.ffn_blocks = nn.ModuleList(
            [FFN_block(embed_dim = embed_dim, hidden_dim = hidden_dim, act=act)
            for _ in range(depth)])
    def forward(self, x):
        x = self.fc_in(x)
        for b in self.ffn_blocks:
             x= x + b(x)
        return self.fc_out(x)

def average_gradients(layers, depth):
    # Average gradients accumulated from repeated block applications.
    for p in layers.parameters():
        p.register_hook(
            lambda grad, depth=depth: grad / depth
        )
