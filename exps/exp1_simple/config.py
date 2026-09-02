# config.py
import torch
from models import *
import numpy as np
# =========================
# Reproducibility
# =========================
seed = 42

# =========================
# Device
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Data Hyperparameters
# =========================
DATA_ROOT = "./data"
batch_size = 512
N_SAMPLES = 10000 * 6
TRAIN_SPLIT = 0.75
num_workers = 4



# =========================
# Training Hyperparameters
# =========================
EPOCHS = 2**10 * 2

# =========================
# Model Hyperparameters
# =========================
data_dim = 1
input_dim = data_dim
depth = 8

embed_dim = 64
hidden_dim = embed_dim * 4

output_dim = data_dim

# base_LR = 1 / hidden_dim / depth / 8
base_LR = 1e-5

# =========================
# Experiment Registry
# =========================
model_configs = []

def register_model(model_class, act, LR,embed_dim = embed_dim, hidden_dim = hidden_dim, composed_proj = True):
    model_configs.append({
        "name": f"{model_class.__name__}",
        "class": model_class,
        "kwargs": dict(
            input_dim = input_dim,
            embed_dim = embed_dim,
            hidden_dim = hidden_dim,
            depth = depth,
            output_dim = output_dim,
            act=act,
            composed_proj = composed_proj,
    ),
        "adaptive": False,
        "lr": LR,
        "epochs": EPOCHS,
    })
register_model(ResNet_rnb_4L, "gelu", LR = base_LR,   composed_proj = True)
register_model(ResNet_rnb, "gelu", LR = base_LR,   composed_proj = True)
register_model(ResNet_rnb_4m, "gelu", LR = base_LR,   composed_proj = True)
register_model(ResNet_classical, "gelu", LR = base_LR,  composed_proj = True)
register_model(ResNet_classical_qtr_depth, "gelu", LR = base_LR,  composed_proj = True)
register_model(ResNet_classical_qtr_width, "gelu", LR = base_LR,  composed_proj = True)
register_model(ResNet_one_same, "gelu", LR = base_LR,  composed_proj = True)


