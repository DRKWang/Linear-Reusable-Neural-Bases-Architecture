# models.py
import torch
from baseline import *
from model import *
# from model_qkv_nueron import *
# from baseline_new_attn import *
# from baseline_linear_attn import *
from model_qkvo_nueron_more_attn import *
# from model_qkvo_nueron_pll import *

# =========================
# Global defaults
# =========================
EPOCHS = 800

#this will also include the spe tokens
TOT_VOCAB_SIZE = 2**14
SEQ_LEN = 128

EMBED_DIM = 512
depth = 8
num_heads = 8
DROPOUT = 0.00
hid_dim =  EMBED_DIM * 4
# =========================
# Model config registry
# =========================
model_configs = []

# LR = 1 /2 * 1/ (hid_dim * depth)

# LR = 1e-5
LR = 1/ hid_dim / depth / 4
# LR = 4e-4

def add_models(model, activation, hidden_dim, LR = LR ):
    model_configs.append(
        {
            "name": f"{model.__name__}",
            "class": model,
            "kwargs": dict(
                vocab_size=TOT_VOCAB_SIZE,
                seq_len=SEQ_LEN,
                embed_dim=EMBED_DIM,
                depth=depth,
                num_heads=num_heads,
                ff_hidden_dim=hidden_dim,
                dropout=DROPOUT,
                activation=activation,
            ),
            "lr": LR,
            "epochs": EPOCHS,
        }
    )

# =========================
# Register experiments
# =========================
# add_models(TF_new_attn, "gelu", hidden_dim = hid_dim, LR = LR)
add_models(TF_rnb_qkvo_ffn_8h8m, "gelu", hidden_dim = hid_dim, LR = LR )
add_models(TF_classical, "gelu", hidden_dim = hid_dim, LR = LR)
add_models(TF_rnb_qkvo_ffn_6h6m, "gelu", hidden_dim = hid_dim, LR = LR )
add_models(TF_rnb_qkvo_ffn_4h4m, "gelu", hidden_dim = hid_dim, LR = LR )
add_models(TF_rnb_qkvo_ffn_2h2m, "gelu", hidden_dim = hid_dim, LR = LR)

# add_models(TF_rnb_qkvo_ffn_4pll, "gelu", hidden_dim = hid_dim, LR = LR )

# add_models(TF_lin_attn, "gelu", hidden_dim = hid_dim, LR = LR)
# add_models(TF_same_block, "gelu", hidden_dim = hid_dim, LR = LR)
# add_models(TF_rnb_ffn, "gelu", hidden_dim = hid_dim, LR = LR)
