#!/usr/bin/env python
# coding: utf-8

# =========================
# Imports
# =========================
import os
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import *
from datamaker import (
    build_dataloader,
    build_dataset,
    ground_truth_function,
)


font_size = 14


# ===========================================================
# Matplotlib setup
# ===========================================================
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "lines.linewidth": 2.0,
    "pdf.fonttype": 42,
})


# =========================
# Output folders
# =========================
CKPT_DIR = "checkpoints"
FIG_DIR = "figs"

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ===========================================================
# Utilities
# ===========================================================
def synchronize_device():
    """
    Synchronize CUDA before measuring time.

    CUDA operations are asynchronous, so synchronization is
    needed to obtain accurate GPU timing.
    """
    if (
        torch.cuda.is_available()
        and str(DEVICE).startswith("cuda")
    ):
        torch.cuda.synchronize()


def print_model_info(model):
    """
    Print the model architecture, parameter shapes,
    and parameter counts.
    """
    print("=" * 80)
    print(model)
    print("=" * 80)

    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()

        total_params += num_params

        if param.requires_grad:
            trainable_params += num_params

        print(
            f"{name:45s} "
            f"{str(tuple(param.shape)):20s} "
            f"{num_params:12,}"
        )

    print("=" * 80)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(
        f"Non-trainable:        "
        f"{total_params - trainable_params:,}"
    )
    print("=" * 80)


def count_params(model):
    """
    Count the trainable parameters of a model.
    """
    return sum(
        param.numel()
        for param in model.parameters()
        if param.requires_grad
    )


def make_model_name(cfg, param_count):
    """
    Construct the model name used for checkpoints and plots.
    """
    name_str = str(
        cfg.get("name", "model")
    )

    name = (
        f"{name_str}"
        f"--para={param_count / 1000:.1f}K"
    )

    return name


def show_in_greek(name: str) -> str:
    """
    Replace selected substrings with LaTeX symbols
    for plotting.
    """
    return name.replace(
        "_pi_",
        r"$\pi$",
    )


# ===========================================================
# Evaluation
# ===========================================================
@torch.no_grad()
def evaluate(model, data_loader):
    """
    Evaluate a model on a dataset using mean squared error.

    The returned value is the sample-weighted average MSE
    over the complete dataset.
    """
    model.eval()

    criterion = nn.MSELoss()

    total_loss = 0.0
    total_samples = 0

    for x, y in data_loader:
        x = x.to(
            DEVICE,
            non_blocking=True,
        )

        y = y.to(
            DEVICE,
            non_blocking=True,
        )

        predictions = model(x)

        loss = criterion(
            predictions,
            y,
        )

        batch_size = x.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    average_mse = (
        total_loss / max(total_samples, 1)
    )

    return average_mse


# ===========================================================
# Training
# ===========================================================
def train_model(cfg, train_ds):
    # -------------------------------------------------------
    # Build model
    # -------------------------------------------------------
    model = cfg["class"](
        **cfg["kwargs"]
    ).to(DEVICE)

    print_model_info(model)

    param_count = count_params(model)

    model_name = make_model_name(
        cfg,
        param_count,
    )

    ckpt_path = os.path.join(
        CKPT_DIR,
        model_name + ".pt",
    )

    train_loader = build_dataloader(
        train_ds
    )

    # -------------------------------------------------------
    # Optimizer and loss
    # -------------------------------------------------------
    base_lr = float(
        cfg["lr"]
    )

    print(f"Model:   {model_name}")
    print(f"Base LR: {base_lr:.6e}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(
            cfg.get(
                "weight_decay",
                0.0,
            )
        ),
    )

    criterion = nn.MSELoss()

    # -------------------------------------------------------
    # Load existing checkpoint
    # -------------------------------------------------------
    if os.path.exists(ckpt_path):
        print(
            f"\nLoading existing model: "
            f"{model_name}"
        )

        checkpoint = torch.load(
            ckpt_path,
            map_location=DEVICE,
        )

        model.load_state_dict(
            checkpoint["model_state"]
        )

        train_losses = checkpoint.get(
            "train_losses",
            [],
        )

        epoch_times = checkpoint.get(
            "epoch_times",
            [],
        )

        cumulative_times = checkpoint.get(
            "cumulative_times",
            list(np.cumsum(epoch_times)),
        )

        return {
            "name": model_name,
            "model": model,
            "losses": train_losses,
            "times": epoch_times,
            "cumulative_times": cumulative_times,
            "param_count": checkpoint.get(
                "param_count",
                param_count,
            ),
        }

    # -------------------------------------------------------
    # Training history
    # -------------------------------------------------------
    train_losses = []
    epoch_times = []
    cumulative_times = []

    minimal_loss = float("inf")
    minimal_state = None
    best_epoch = -1

    total_training_time = 0.0

    progress_bar = tqdm(
        range(cfg["epochs"]),
        desc=model_name,
    )

    # -------------------------------------------------------
    # Epoch loop
    # -------------------------------------------------------
    for epoch in progress_bar:
        synchronize_device()

        epoch_start_time = time.perf_counter()

        model.train()

        running_loss = 0.0
        total_samples = 0

        for x, y in train_loader:
            x = x.to(
                DEVICE,
                non_blocking=True,
            )

            y = y.to(
                DEVICE,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            predictions = model(x)

            loss = criterion(
                predictions,
                y,
            )

            loss.backward()

            optimizer.step()

            batch_size = x.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

        synchronize_device()

        epoch_time = (
            time.perf_counter()
            - epoch_start_time
        )

        average_loss = (
            running_loss
            / max(total_samples, 1)
        )

        total_training_time += epoch_time

        train_losses.append(
            average_loss
        )

        epoch_times.append(
            epoch_time
        )

        cumulative_times.append(
            total_training_time
        )

        # ---------------------------------------------------
        # Optional adaptive learning rate
        # ---------------------------------------------------
        if cfg.get("adaptive", False):
            new_lr = base_lr * max(
                np.sqrt(average_loss),
                1e-8,
            )

            for param_group in optimizer.param_groups:
                param_group["lr"] = new_lr

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        # ---------------------------------------------------
        # Track best model according to training loss
        # ---------------------------------------------------
        if average_loss < minimal_loss:
            minimal_loss = average_loss
            best_epoch = epoch

            minimal_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

        # ---------------------------------------------------
        # Progress-bar information
        # ---------------------------------------------------
        progress_bar.set_postfix(
            loss=f"{average_loss:.4e}",
            best=f"{minimal_loss:.4e}",
            lr=f"{current_lr:.2e}",
            epoch_time=f"{epoch_time:.3f}s",
            total_time=(
                f"{total_training_time:.1f}s"
            ),
        )

    # -------------------------------------------------------
    # Restore best model
    # -------------------------------------------------------
    if minimal_state is not None:
        model.load_state_dict(
            minimal_state
        )

    # -------------------------------------------------------
    # Save checkpoint
    # -------------------------------------------------------
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),

        "train_losses": train_losses,
        "epoch_times": epoch_times,
        "cumulative_times": cumulative_times,

        "param_count": param_count,
        "best_epoch": best_epoch,
        "minimal_loss": minimal_loss,

        "cfg": cfg,
    }

    torch.save(
        checkpoint,
        ckpt_path,
    )

    print(f"\nSaved model to {ckpt_path}")
    print(f"Best epoch: {best_epoch + 1}")
    print(
        f"Best training MSE: "
        f"{minimal_loss:.8e}"
    )
    print(
        f"Total training time: "
        f"{total_training_time:.4f} seconds"
    )

    return {
        "name": model_name,
        "model": model,
        "losses": train_losses,
        "times": epoch_times,
        "cumulative_times": cumulative_times,
        "param_count": param_count,
    }


# ===========================================================
# Plot training loss against epoch
# ===========================================================
def plot_training(results):
    """
    Plot the training MSE against the training epoch.
    """
    plt.figure(figsize=(8, 5))

    for result in results:
        losses = np.asarray(
            result["losses"],
            dtype=float,
        )

        if len(losses) == 0:
            continue

        epochs = np.arange(
            1,
            len(losses) + 1,
        )

        plt.plot(
            epochs,
            losses,
            label=show_in_greek(
                result["name"]
            ),
        )

    plt.xlabel("Epoch")
    plt.ylabel("Training MSE Loss")
    plt.yscale("log")

    plt.legend(
        fontsize=font_size
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIG_DIR,
            "training_loss.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ===========================================================
# Plot test MSE for all trained models
# ===========================================================
def plot_test_mse(results):
    """
    Plot the MSE loss of each trained model on the test dataset.

    Lower test MSE indicates better predictive performance.
    """
    valid_results = [
        result
        for result in results
        if (
            "test_loss" in result
            and np.isfinite(result["test_loss"])
            and result["test_loss"] > 0
        )
    ]

    if len(valid_results) == 0:
        print(
            "No positive finite test MSE "
            "values are available."
        )
        return

    # Sort models from lowest test MSE to highest test MSE.
    sorted_results = sorted(
        valid_results,
        key=lambda result: result["test_loss"],
    )

    model_names = [
        show_in_greek(result["name"])
        for result in sorted_results
    ]

    test_mse_values = np.asarray(
        [
            result["test_loss"]
            for result in sorted_results
        ],
        dtype=float,
    )

    positions = np.arange(
        len(sorted_results)
    )

    figure_width = max(
        8,
        1.5 * len(sorted_results),
    )

    plt.figure(
        figsize=(
            figure_width,
            6,
        )
    )

    bars = plt.bar(
        positions,
        test_mse_values,
        width=0.65,
    )

    plt.xticks(
        positions,
        model_names,
        rotation=20,
        ha="right",
    )

    # plt.xlabel("Model")
    plt.ylabel("Test MSE Loss")
    # plt.title(
    #     "Test MSE of Trained Models"
    # )

    # Use a logarithmic scale because model errors may
    # differ by several orders of magnitude.
    plt.yscale("log")

    # Leave some space above the largest bar for labels.
    minimum_mse = np.min(
        test_mse_values
    )

    maximum_mse = np.max(
        test_mse_values
    )

    plt.ylim(
        minimum_mse * 0.5,
        maximum_mse * 2.5,
    )

    # Display each test MSE above its bar.
    for bar, mse in zip(
        bars,
        test_mse_values,
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            mse * 1.08,
            f"{mse:.2e}",
            ha="center",
            va="bottom",
            fontsize=font_size - 1,
        )

    plt.tight_layout()

    save_path = os.path.join(
        FIG_DIR,
        "test_mse.pdf",
    )

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved test MSE plot to "
        f"{save_path}"
    )


# ===========================================================
# Plot time required for each epoch
# ===========================================================
def plot_time_per_epoch(results):
    """
    Plot the training time required by each individual epoch.
    """
    plt.figure(figsize=(8, 5))

    for result in results:
        epoch_times = np.asarray(
            result["times"],
            dtype=float,
        )

        if len(epoch_times) == 0:
            continue

        epochs = np.arange(
            1,
            len(epoch_times) + 1,
        )

        plt.plot(
            epochs,
            epoch_times,
            label=show_in_greek(
                result["name"]
            ),
        )

    plt.xlabel("Epoch")
    plt.ylabel(
        "Training Time per Epoch (seconds)"
    )

    plt.legend(
        fontsize=font_size
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIG_DIR,
            "training_time_per_epoch.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ===========================================================
# Plot cumulative training time against epoch
# ===========================================================
def plot_cumulative_training_time(results):
    """
    Plot the cumulative training time as the epoch increases.
    """
    plt.figure(figsize=(8, 5))

    for result in results:
        cumulative_times = np.asarray(
            result["cumulative_times"],
            dtype=float,
        )

        if len(cumulative_times) == 0:
            continue

        epochs = np.arange(
            1,
            len(cumulative_times) + 1,
        )

        plt.plot(
            epochs,
            cumulative_times,
            label=show_in_greek(
                result["name"]
            ),
        )

    plt.xlabel("Epoch")
    plt.ylabel(
        "Cumulative Training Time (seconds)"
    )

    plt.legend(
        fontsize=font_size
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIG_DIR,
            "cumulative_training_time.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ===========================================================
# Plot smoothed time per epoch
# ===========================================================
def plot_smoothed_training_time(
    results,
    window_size=10,
):
    """
    Plot a moving average of the per-epoch training time.

    This can make the timing comparison easier to read
    because individual epoch times may fluctuate.
    """
    plt.figure(figsize=(8, 5))

    for result in results:
        epoch_times = np.asarray(
            result["times"],
            dtype=float,
        )

        if len(epoch_times) == 0:
            continue

        current_window = min(
            window_size,
            len(epoch_times),
        )

        kernel = (
            np.ones(current_window)
            / current_window
        )

        smoothed_times = np.convolve(
            epoch_times,
            kernel,
            mode="valid",
        )

        epochs = np.arange(
            current_window,
            len(epoch_times) + 1,
        )

        plt.plot(
            epochs,
            smoothed_times,
            label=show_in_greek(
                result["name"]
            ),
        )

    plt.xlabel("Epoch")

    plt.ylabel(
        f"Average Training Time per Epoch (s)\n"
        f"({window_size}-Epoch Moving Average)"
    )

    plt.legend(
        fontsize=font_size
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIG_DIR,
            "smoothed_training_time.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ===========================================================
# Plot all models against ground truth
# ===========================================================
def plot_all_models_vs_gt(
    results,
    x_ranges=(-1.2, 1.2),
):
    """
    Plot all trained models together with the ground-truth
    function.
    """
    plt.figure(figsize=(9, 6))

    x = torch.linspace(
        x_ranges[0],
        x_ranges[1],
        1200,
    ).unsqueeze(1).to(DEVICE)

    with torch.no_grad():
        y_ground_truth = (
            ground_truth_function(x)
        )

    x_numpy = (
        x.detach()
        .cpu()
        .numpy()
    )

    ground_truth_numpy = (
        y_ground_truth.detach()
        .cpu()
        .numpy()
    )

    plt.plot(
        x_numpy,
        ground_truth_numpy,
        linewidth=3,
        color="black",
        label="Ground Truth",
    )

    for result in results:
        model = result["model"]

        model.eval()

        with torch.no_grad():
            y_prediction = model(x)

        prediction_numpy = (
            y_prediction.detach()
            .cpu()
            .numpy()
        )

        plt.plot(
            x_numpy,
            prediction_numpy,
            linewidth=2,
            alpha=0.85,
            label=(
                f"{show_in_greek(result['name'])} "
                f"({result['param_count'] // 1000}K)"
            ),
        )

    plt.xlim(
        x_ranges[0],
        x_ranges[1],
    )

    plt.xlabel("x")
    plt.ylabel("f(x)")

    # plt.title(
    #     "Model Comparison vs. Ground Truth"
    # )

    plt.legend(
        fontsize=font_size
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            FIG_DIR,
            "all_models_vs_gt.pdf",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()


# ===========================================================
# Print timing comparison
# ===========================================================
def print_timing_results(results):
    """
    Print total, average, standard deviation, minimum,
    and maximum epoch times for every model.
    """
    print(
        "\n===== TRAINING TIME COMPARISON =====\n"
    )

    results_sorted = sorted(
        results,
        key=lambda result: (
            np.sum(result["times"])
            if len(result["times"]) > 0
            else float("inf")
        ),
    )

    for rank, result in enumerate(
        results_sorted,
        start=1,
    ):
        epoch_times = np.asarray(
            result["times"],
            dtype=float,
        )

        if len(epoch_times) == 0:
            continue

        total_time = np.sum(epoch_times)
        mean_time = np.mean(epoch_times)
        std_time = np.std(epoch_times)
        minimum_time = np.min(epoch_times)
        maximum_time = np.max(epoch_times)

        print(
            f"{rank:02d}. {result['name']}\n"
            f"    Total training time:  "
            f"{total_time:.6f} seconds\n"
            f"    Mean time per epoch:  "
            f"{mean_time:.6f} seconds\n"
            f"    Std. time per epoch:  "
            f"{std_time:.6f} seconds\n"
            f"    Minimum epoch time:   "
            f"{minimum_time:.6f} seconds\n"
            f"    Maximum epoch time:   "
            f"{maximum_time:.6f} seconds\n"
        )


# ===========================================================
# Main
# ===========================================================
def main():
    results = []

    print(
        "\n===== TRAINING PHASE =====\n"
    )

    train_ds, test_ds = build_dataset()

    print(
        f"Training samples: "
        f"{len(train_ds):,}"
    )

    print(
        f"Test samples:     "
        f"{len(test_ds):,}"
    )

    # -------------------------------------------------------
    # Train all models
    # -------------------------------------------------------
    for cfg in model_configs:
        result = train_model(
            cfg=cfg,
            train_ds=train_ds,
        )

        results.append(result)

    # -------------------------------------------------------
    # Evaluate all models on the test dataset
    # -------------------------------------------------------
    print(
        "\n===== EVALUATION PHASE =====\n"
    )

    test_loader = build_dataloader(
        test_ds
    )

    for result in results:
        test_mse = evaluate(
            model=result["model"],
            data_loader=test_loader,
        )

        result["test_loss"] = test_mse

    # -------------------------------------------------------
    # Sort models by test MSE
    # -------------------------------------------------------
    results_sorted = sorted(
        results,
        key=lambda result: result["test_loss"],
    )

    print(
        "\n===== SORTED TEST MSE RESULTS =====\n"
    )

    for rank, result in enumerate(
        results_sorted,
        start=1,
    ):
        print(
            f"{rank:02d}. {result['name']}\n"
            f"    Parameters: "
            f"{result['param_count']:,}\n"
            f"    Test MSE:   "
            f"{result['test_loss']:.8e}\n"
        )

    # -------------------------------------------------------
    # Print training-time comparison
    # -------------------------------------------------------
    print_timing_results(results)

    # -------------------------------------------------------
    # Generate figures
    # -------------------------------------------------------

    # Training MSE against epoch.
    plot_training(results)

    # Final test MSE comparison for all models.
    plot_test_mse(results)

    # Time required by each individual epoch.
    plot_time_per_epoch(results)

    # Total elapsed training time against epoch.
    plot_cumulative_training_time(results)

    # Moving-average training-time curve.
    plot_smoothed_training_time(
        results,
        window_size=10,
    )

    # Model predictions against the ground truth.
    plot_all_models_vs_gt(results)

    print(
        "\nAll figures saved in ./figs/"
    )

    print(
        "All checkpoints saved in ./checkpoints/"
    )


if __name__ == "__main__":
    main()