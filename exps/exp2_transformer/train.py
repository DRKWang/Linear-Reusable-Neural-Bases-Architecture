#!/usr/bin/env python
# coding: utf-8

# =========================
# Imports
# =========================
import os
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import model_configs, SEQ_LEN
from datamaker import (
    BASE_SEED,
    BATCH_SIZE,
    NUM_TEST_BATCHES,
    NUM_TRAIN_BATCHES,
    load_wikitext2_raw,
    build_tokenizer,
    dataset_maker,
)


# =========================
# Plot configuration
# =========================
font_size = 14

plt.rcParams.update({
    "font.size": font_size,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.3,
    "lines.linewidth": 2.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# =========================
# Global configuration
# =========================
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

CKPT_DIR = "checkpoints"
FIG_DIR = "figs"

os.makedirs(
    CKPT_DIR,
    exist_ok=True,
)

os.makedirs(
    FIG_DIR,
    exist_ok=True,
)


# =========================
# Reproducibility
# =========================
torch.manual_seed(
    BASE_SEED
)

random.seed(
    BASE_SEED
)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(
        BASE_SEED
    )


# =========================
# Load dataset and tokenizer
# =========================
raw_dataset = load_wikitext2_raw()

tokenizer = build_tokenizer(
    dataset=raw_dataset,
)


# =========================
# Utilities
# =========================
def count_trainable_params(model):
    """
    Count the number of trainable model parameters.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def make_model_name(
    cfg,
    param_count,
):
    """
    Construct a descriptive model name.
    """
    base_name = cfg.get(
        "name",
        cfg["class"].__name__,
    )

    parts = [
        base_name,
        f"Para={param_count / 1_000_000:.2f}M",
    ]

    return "--".join(parts)


def synchronize_device():
    """
    Synchronize CUDA operations for accurate timing.

    CUDA operations are asynchronous, so synchronization
    is needed immediately before starting and stopping
    a timer.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def build_random_loader(
    dataset,
    batch_size,
    num_batches,
    seed,
    replacement=True,
):
    """
    Build a DataLoader that draws exactly num_batches
    random batches.

    Parameters
    ----------
    dataset:
        Dataset from which token windows are sampled.

    batch_size:
        Number of samples in each batch.

    num_batches:
        Number of batches returned by the DataLoader.

    seed:
        Random seed used by the sampler.

    replacement:
        If True, the same token window may be sampled
        multiple times.
    """
    if num_batches is None:
        raise ValueError(
            "num_batches must not be None."
        )

    if num_batches <= 0:
        raise ValueError(
            "num_batches must be positive."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    num_samples = (
        batch_size * num_batches
    )

    if (
        not replacement
        and num_samples > len(dataset)
    ):
        raise ValueError(
            "Cannot sample without replacement because "
            f"num_samples={num_samples} exceeds "
            f"dataset size={len(dataset)}."
        )

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    sampler = RandomSampler(
        dataset,
        replacement=replacement,
        num_samples=num_samples,
        generator=generator,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    return loader


# =========================
# Evaluation
# =========================
@torch.no_grad()
def evaluate_model(
    model,
    test_dataset,
    batch_size=BATCH_SIZE,
    num_batches=NUM_TEST_BATCHES,
    device=DEVICE,
    seed=BASE_SEED + 9999,
):
    """
    Evaluate the model using cross-entropy loss.

    The same seed can be used across epochs and models
    so that all models are evaluated using the same
    randomly sampled test windows.
    """
    model.eval()

    test_loader = build_random_loader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_batches=num_batches,
        seed=seed,
        replacement=True,
    )

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    actual_batches = 0

    for x, y in test_loader:
        x = x.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        logits = model(x)

        ce_loss = criterion(
            logits.reshape(
                -1,
                logits.size(-1),
            ),
            y.reshape(-1),
        )

        total_loss += ce_loss.item()
        actual_batches += 1

    if actual_batches == 0:
        raise RuntimeError(
            "The evaluation DataLoader produced "
            "no batches."
        )

    average_loss = (
        total_loss / actual_batches
    )

    return average_loss


# =========================
# Text generation
# =========================
@torch.no_grad()
def generate_text(
    model,
    seq_len,
    prompt,
    tokenizer,
    max_new_tokens=300,
    temperature=0.8,
    top_k=50,
    device=DEVICE,
):
    """
    Generate text autoregressively from a prompt.
    """
    model.eval()

    if temperature <= 0:
        raise ValueError(
            "temperature must be greater than zero."
        )

    encoded = tokenizer.encode(
        prompt
    )

    ids = encoded.ids

    if len(ids) == 0:
        raise ValueError(
            "The prompt produced no tokens."
        )

    pad_token_id = tokenizer.token_to_id(
        "[PAD]"
    )

    if pad_token_id is None:
        pad_token_id = 0

    x = torch.tensor(
        ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    for _ in range(max_new_tokens):
        x_cond_raw = x[
            :,
            -seq_len:,
        ]

        if x_cond_raw.size(1) < seq_len:
            pad_len = (
                seq_len
                - x_cond_raw.size(1)
            )

            pad = torch.full(
                (
                    x_cond_raw.size(0),
                    pad_len,
                ),
                pad_token_id,
                dtype=torch.long,
                device=device,
            )

            x_cond = torch.cat(
                [
                    pad,
                    x_cond_raw,
                ],
                dim=1,
            )

        else:
            x_cond = x_cond_raw

        logits = model(
            x_cond
        )

        logits = logits[
            :,
            -1,
            :,
        ]

        logits = (
            logits / temperature
        )

        if top_k is not None:
            if top_k <= 0:
                raise ValueError(
                    "top_k must be positive or None."
                )

            effective_top_k = min(
                top_k,
                logits.size(-1),
            )

            values, _ = torch.topk(
                logits,
                effective_top_k,
            )

            threshold = values[
                :,
                -1,
            ].unsqueeze(-1)

            logits = torch.where(
                logits < threshold,
                torch.full_like(
                    logits,
                    float("-inf"),
                ),
                logits,
            )

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

        next_id = torch.multinomial(
            probabilities,
            num_samples=1,
        )

        x = torch.cat(
            [
                x,
                next_id,
            ],
            dim=1,
        )

    output_ids = x[
        0
    ].tolist()

    return tokenizer.decode(
        output_ids
    )


# =========================
# Training
# =========================
def train_model(
    cfg,
    train_dataset,
    test_dataset,
):
    """
    Train one model and record:

    - training CE loss per epoch,
    - test CE loss per epoch,
    - training time per epoch,
    - average training time per epoch.
    """
    epochs = cfg["epochs"]

    model = cfg["class"](
        **cfg["kwargs"]
    ).to(DEVICE)

    param_count = count_trainable_params(
        model
    )

    model_name = make_model_name(
        cfg,
        param_count,
    )

    ckpt_path = os.path.join(
        CKPT_DIR,
        model_name + ".pt",
    )

    base_lr = cfg["lr"]

    print(
        f"\nModel: {model_name}"
    )

    print(
        f"Base LR: {base_lr:.2e}"
    )

    print(
        f"Device:  {DEVICE}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
    )

    criterion = nn.CrossEntropyLoss()

    # ==================================================
    # Reload checkpoint
    # ==================================================
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

        train_loss = checkpoint.get(
            "train_loss",
            [],
        )

        test_loss = checkpoint.get(
            "test_loss",
            [],
        )

        epoch_times = checkpoint.get(
            "epoch_times",
            [],
        )

        if len(epoch_times) > 0:
            average_epoch_time = (
                sum(epoch_times)
                / len(epoch_times)
            )

            total_training_time = sum(
                epoch_times
            )

        else:
            average_epoch_time = float(
                "nan"
            )

            total_training_time = float(
                "nan"
            )

        return {
            "name": model_name,
            "model": model,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "epoch_times": epoch_times,
            "average_epoch_time": (
                average_epoch_time
            ),
            "total_training_time": (
                total_training_time
            ),
            "param_count": param_count,
        }

    # ==================================================
    # Metric histories
    # ==================================================
    train_loss = []
    test_loss = []
    epoch_times = []

    progress_bar = tqdm(
        range(epochs),
        desc=model_name,
        dynamic_ncols=True,
    )

    for epoch in progress_bar:
        model.train()

        running_ce_loss = 0.0
        actual_batches = 0

        # Draw a new reproducible random collection
        # of windows during each epoch.
        train_loader = build_random_loader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            num_batches=NUM_TRAIN_BATCHES,
            seed=BASE_SEED + epoch,
            replacement=True,
        )

        # Synchronize immediately before starting
        # the training timer.
        synchronize_device()

        epoch_start_time = (
            time.perf_counter()
        )

        # ==================================================
        # Training batches
        # ==================================================
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

            logits = model(
                x
            )

            ce_loss = criterion(
                logits.reshape(
                    -1,
                    logits.size(-1),
                ),
                y.reshape(-1),
            )

            ce_loss.backward()

            # Optional gradient clipping:
            #
            # torch.nn.utils.clip_grad_norm_(
            #     model.parameters(),
            #     max_norm=1.0,
            # )

            optimizer.step()

            running_ce_loss += (
                ce_loss.item()
            )

            actual_batches += 1

        if actual_batches == 0:
            raise RuntimeError(
                "The training DataLoader produced "
                "no batches."
            )

        # Synchronize before stopping the timer.
        synchronize_device()

        epoch_end_time = (
            time.perf_counter()
        )

        epoch_time = (
            epoch_end_time
            - epoch_start_time
        )

        avg_train_ce_loss = (
            running_ce_loss
            / actual_batches
        )

        # ==================================================
        # Test-set evaluation
        # ==================================================
        avg_test_ce_loss = evaluate_model(
            model=model,
            test_dataset=test_dataset,
            batch_size=BATCH_SIZE,
            num_batches=NUM_TEST_BATCHES,
            device=DEVICE,
            seed=BASE_SEED + 9999,
        )
        # avg_test_ce_loss = 0

        train_loss.append(
            avg_train_ce_loss
        )

        test_loss.append(
            avg_test_ce_loss
        )

        epoch_times.append(
            epoch_time
        )

        running_average_time = (
            sum(epoch_times)
            / len(epoch_times)
        )

        progress_bar.set_postfix(
            train_ce=(
                f"{avg_train_ce_loss:.6f}"
            ),
            test_ce=(
                f"{avg_test_ce_loss:.6f}"
            ),
            epoch_time=(
                f"{epoch_time:.3f}s"
            ),
            avg_time=(
                f"{running_average_time:.3f}s"
            ),
        )

        # ==================================================
        # Text-generation test
        # ==================================================
        if epoch % 40 == 0:
            print(
                "\n--- Model Text Generation Test ---"
            )

            prompt = "The game"

            text = generate_text(
                model=model,
                seq_len=cfg["kwargs"].get(
                    "seq_len",
                    SEQ_LEN,
                ),
                prompt=prompt,
                tokenizer=tokenizer,
                max_new_tokens=30,
                temperature=0.8,
                top_k=1,
                device=DEVICE,
            )

            print(
                f"Generated text: {text}\n"
                "-----------------------------------\n"
            )

        if epoch % 1000 == 0:
            print(
                f"Epoch {epoch}: "
                f"train CE loss = "
                f"{avg_train_ce_loss:.6f}, "
                f"test CE loss = "
                f"{avg_test_ce_loss:.6f}, "
                f"training time = "
                f"{epoch_time:.3f} seconds"
            )

    average_epoch_time = (
        sum(epoch_times)
        / len(epoch_times)
    )

    total_training_time = sum(
        epoch_times
    )

    # ==================================================
    # Save checkpoint
    # ==================================================
    torch.save(
        {
            "model_state": (
                model.state_dict()
            ),
            "train_loss": train_loss,
            "test_loss": test_loss,
            "epoch_times": epoch_times,
            "average_epoch_time": (
                average_epoch_time
            ),
            "total_training_time": (
                total_training_time
            ),
            "cfg": cfg,
        },
        ckpt_path,
    )

    return {
        "name": model_name,
        "model": model,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "epoch_times": epoch_times,
        "average_epoch_time": (
            average_epoch_time
        ),
        "total_training_time": (
            total_training_time
        ),
        "param_count": param_count,
    }


# =========================
# Plot training losses
# =========================
def plot_training_losses(
    results,
    output_path=os.path.join(
        FIG_DIR,
        "train_loss.pdf",
    ),
):
    """
    Plot training CE loss against epoch.
    """
    plt.figure(
        figsize=(9, 5)
    )

    plotted_any_curve = False

    for result in results:
        losses = result.get(
            "train_loss",
            [],
        )

        if len(losses) == 0:
            print(
                f"Skipping training-loss curve for "
                f"{result['name']}: no history found."
            )

            continue

        epochs = range(
            1,
            len(losses) + 1,
        )

        plt.plot(
            epochs,
            losses,
            label=result["name"],
        )

        plotted_any_curve = True

    if not plotted_any_curve:
        print(
            "No training-loss histories are available."
        )

        plt.close()
        return

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Training CE Loss"
    )

    plt.legend(
        fontsize=font_size,
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()


# =========================
# Plot test losses
# =========================
def plot_test_losses(
    results,
    output_path=os.path.join(
        FIG_DIR,
        "test_loss.pdf",
    ),
):
    """
    Plot test CE loss against epoch.
    """
    plt.figure(
        figsize=(9, 5)
    )

    plotted_any_curve = False

    for result in results:
        losses = result.get(
            "test_loss",
            [],
        )

        if len(losses) == 0:
            print(
                f"Skipping test-loss curve for "
                f"{result['name']}: no history found."
            )

            continue

        epochs = range(
            1,
            len(losses) + 1,
        )

        plt.plot(
            epochs,
            losses,
            label=result["name"],
        )

        plotted_any_curve = True

    if not plotted_any_curve:
        print(
            "No test-loss histories are available."
        )

        plt.close()
        return

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Test CE Loss"
    )

    plt.legend(
        fontsize=font_size,
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()


# =========================
# Plot train and test losses
# =========================
def plot_train_test_losses(
    results,
    output_path=os.path.join(
        FIG_DIR,
        "train_test_loss.pdf",
    ),
):
    """
    Plot both training and test losses in one figure.

    Solid lines:
        Training losses.

    Dashed lines:
        Test losses.
    """
    plt.figure(
        figsize=(10, 6)
    )

    plotted_any_curve = False

    for result in results:
        train_losses = result.get(
            "train_loss",
            [],
        )

        test_losses = result.get(
            "test_loss",
            [],
        )

        if len(train_losses) > 0:
            train_epochs = range(
                1,
                len(train_losses) + 1,
            )

            line, = plt.plot(
                train_epochs,
                train_losses,
                label=(
                    f"{result['name']} - Train"
                ),
            )

            plotted_any_curve = True

            if len(test_losses) > 0:
                test_epochs = range(
                    1,
                    len(test_losses) + 1,
                )

                plt.plot(
                    test_epochs,
                    test_losses,
                    linestyle="--",
                    color=line.get_color(),
                    label=(
                        f"{result['name']} - Test"
                    ),
                )

    if not plotted_any_curve:
        print(
            "No train/test-loss histories "
            "are available."
        )

        plt.close()
        return

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Cross-Entropy Loss"
    )

    plt.legend(
        fontsize=font_size - 2,
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()


# =========================
# Plot epoch training times
# =========================
def plot_epoch_times(
    results,
    output_path=os.path.join(
        FIG_DIR,
        "epoch_training_time.pdf",
    ),
):
    """
    Plot training time for each epoch.
    """
    plt.figure(
        figsize=(9, 5)
    )

    plotted_any_curve = False

    for result in results:
        epoch_times = result.get(
            "epoch_times",
            [],
        )

        if len(epoch_times) == 0:
            print(
                f"Skipping epoch-time curve for "
                f"{result['name']}: no history found."
            )

            continue

        epochs = range(
            1,
            len(epoch_times) + 1,
        )

        plt.plot(
            epochs,
            epoch_times,
            label=result["name"],
        )

        plotted_any_curve = True

    if not plotted_any_curve:
        print(
            "No epoch-time histories are available."
        )

        plt.close()
        return

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Training Time per Epoch (seconds)"
    )

    plt.legend(
        fontsize=font_size,
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()


# =========================
# Plot average training times
# =========================
def plot_average_training_times(
    results,
    output_path=os.path.join(
        FIG_DIR,
        "average_training_time.pdf",
    ),
):
    """
    Plot average training time per epoch for each model.
    """
    names = []
    average_times = []

    for result in results:
        epoch_times = result.get(
            "epoch_times",
            [],
        )

        if len(epoch_times) == 0:
            print(
                f"Skipping average time for "
                f"{result['name']}: no timing history."
            )

            continue

        average_time = (
            sum(epoch_times)
            / len(epoch_times)
        )

        names.append(
            result["name"]
        )

        average_times.append(
            average_time
        )

    if len(average_times) == 0:
        print(
            "No average training-time data "
            "are available."
        )

        return

    plt.figure(
        figsize=(10, 5)
    )

    bars = plt.bar(
        names,
        average_times,
    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "Average Training Time per Epoch (seconds)"
    )

    plt.xticks(
        rotation=25,
        ha="right",
    )

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    for bar, average_time in zip(
        bars,
        average_times,
    ):
        plt.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f"{average_time:.3f}",
            ha="center",
            va="bottom",
            fontsize=font_size - 2,
        )

    plt.tight_layout()

    plt.savefig(
        output_path,
        bbox_inches="tight",
    )

    plt.show()
    plt.close()


# =========================
# Print final results
# =========================
def print_final_results(
    results,
    test_dataset,
):
    """
    Evaluate all trained models and print final metrics.
    """
    print(
        "\n===== FINAL RESULTS ====="
    )

    final_results = []

    for result in results:
        final_test_loss = evaluate_model(
            model=result["model"],
            test_dataset=test_dataset,
            batch_size=BATCH_SIZE,
            num_batches=NUM_TEST_BATCHES,
            device=DEVICE,
            seed=BASE_SEED + 9999,
        )

        epoch_times = result.get(
            "epoch_times",
            [],
        )

        if len(epoch_times) > 0:
            average_epoch_time = (
                sum(epoch_times)
                / len(epoch_times)
            )

            total_training_time = sum(
                epoch_times
            )

            average_time_text = (
                f"{average_epoch_time:.6f} seconds"
            )

            total_time_text = (
                f"{total_training_time:.6f} seconds"
            )

        else:
            average_epoch_time = float(
                "nan"
            )

            total_training_time = float(
                "nan"
            )

            average_time_text = (
                "Not available"
            )

            total_time_text = (
                "Not available"
            )

        result["final_test_loss"] = (
            final_test_loss
        )

        result["average_epoch_time"] = (
            average_epoch_time
        )

        result["total_training_time"] = (
            total_training_time
        )

        final_results.append(
            result
        )

        print(
            f"\n{result['name']}\n"
            f"  Params:                  "
            f"{result['param_count']:,}\n"
            f"  Final test CE loss:      "
            f"{final_test_loss:.6e}\n"
            f"  Average time per epoch:  "
            f"{average_time_text}\n"
            f"  Total training time:     "
            f"{total_time_text}\n"
        )

    # Sort from the smallest test loss
    # to the largest test loss.
    final_results.sort(
        key=lambda item: item[
            "final_test_loss"
        ]
    )

    print(
        "\n===== RANKING BY TEST LOSS ====="
    )

    for rank, result in enumerate(
        final_results,
        start=1,
    ):
        print(
            f"{rank}. {result['name']}\n"
            f"   Test CE loss: "
            f"{result['final_test_loss']:.6e}"
        )


# =========================
# Generate final text samples
# =========================
def generate_final_samples(
    results,
):
    """
    Generate one longer text sample from every model.
    """
    print(
        "\n===== TEXT GENERATION ====="
    )

    prompt = (
        "English is a language ."
    )

    for result, cfg in zip(
        results,
        model_configs,
    ):
        model = result["model"]

        seq_len = cfg["kwargs"].get(
            "seq_len",
            SEQ_LEN,
        )

        text = generate_text(
            model=model,
            seq_len=seq_len,
            prompt=prompt,
            tokenizer=tokenizer,
            max_new_tokens=300,
            temperature=0.8,
            top_k=1,
            device=DEVICE,
        )

        print(
            "\n" + "=" * 80
        )

        print(
            f"MODEL: {result['name']}"
        )

        print(
            f"PARAMS: "
            f"{result['param_count']:,}"
        )

        print(
            "-" * 80
        )

        print(
            "PROMPT:"
        )

        print(
            prompt
        )

        print(
            "\nGENERATED:"
        )

        print(
            text
        )


# =========================
# Main
# =========================
def main():
    train_dataset = dataset_maker(
        split="train",
        chunk_size=SEQ_LEN,
    )

    test_dataset = dataset_maker(
        split="test",
        chunk_size=SEQ_LEN,
    )

    results = []

    # ==================================================
    # Train models
    # ==================================================
    for cfg in model_configs:
        result = train_model(
            cfg=cfg,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
        )

        results.append(
            result
        )

    # ==================================================
    # Final test-set evaluation
    # ==================================================
    print_final_results(
        results=results,
        test_dataset=test_dataset,
    )

    # ==================================================
    # Plot results
    # ==================================================
    plot_training_losses(
        results=results,
        output_path=os.path.join(
            FIG_DIR,
            "train_loss.pdf",
        ),
    )

    plot_test_losses(
        results=results,
        output_path=os.path.join(
            FIG_DIR,
            "test_loss.pdf",
        ),
    )

    plot_train_test_losses(
        results=results,
        output_path=os.path.join(
            FIG_DIR,
            "train_test_loss.pdf",
        ),
    )

    plot_epoch_times(
        results=results,
        output_path=os.path.join(
            FIG_DIR,
            "epoch_training_time.pdf",
        ),
    )

    plot_average_training_times(
        results=results,
        output_path=os.path.join(
            FIG_DIR,
            "average_training_time.pdf",
        ),
    )

    # ==================================================
    # Generate final text samples
    # ==================================================
    generate_final_samples(
        results=results,
    )


if __name__ == "__main__":
    main()