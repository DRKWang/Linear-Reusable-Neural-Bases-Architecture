import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from config import *


# =========================
# Dataset Config
# =========================
DIM = data_dim


# =========================
# Set Seed Properly
# =========================

def set_seed():
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Ground Truth Function
# =========================

def ground_truth_function(x):
    """
    x: torch.Tensor, shape [n, dim]

    return:
        y: torch.Tensor, shape [n, dim]
    """
    # return x*2 - torch.floor(x)

    # return torch.exp(5*x)/20 - torch.exp(3*x)/20
    res = torch.cos(2 * torch.pi * torch.cos(2 * torch.pi * x))
    # res = torch.cos(2 * torch.pi * res)
    # for i in range(1, 10):
    #     res += torch.sin(2 * torch.pi * i * x )
    return res

    # return torch.cos(4 * x)
    # # return (x+0.5)**3
    #
    # return torch.exp(4*x)/20
# =========================
# Dataset Builder
# =========================

def build_dataset():
    set_seed()

    # Generate X uniformly from [-1, 1]^DIM
    X = np.random.uniform(-1, 1, size=(N_SAMPLES, DIM))
    #
    # # Center X dimension-wise
    # X_mean = X.mean(axis=0, keepdims=True)
    # X = X - X_mean
    #
    # # Normalize each dimension by its standard deviation
    # X_std = X.std(axis=0, keepdims=True)
    # X = X / (X_std + 1e-12)

    # Convert to torch tensor
    x = torch.tensor(X, dtype=torch.float32)

    # Generate target Y
    y = ground_truth_function(x)
    #
    # # Normalize y
    # y_mean = y.mean(axis=0, keepdims=True)
    # y_std = y.std(axis=0, keepdims=True)
    # y = (y - y_mean) / (y_std + 1e-12)

    dataset = TensorDataset(x, y)

    train_size = int(TRAIN_SPLIT * N_SAMPLES)
    test_size = N_SAMPLES - train_size

    generator = torch.Generator().manual_seed(seed)

    train_ds, test_ds = random_split(
        dataset,
        [train_size, test_size],
        generator=generator,
    )

    return train_ds, test_ds

# =========================
# Dataloaders
# =========================

def build_dataloader(dataset, shuffle=True):
    """
    Build deterministic dataloader.
    """
    generator = torch.Generator().manual_seed(seed)

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )

    return data_loader