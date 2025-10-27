"""Training loop for bilinear modular arithmetic with observability and checkpointing."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import arguably
import torch as th
import torch.nn as nn
from loguru import logger
from tqdm import tqdm
import trackio


@dataclass
class TrainingConfig:
    """Configuration for training bilinear modular arithmetic."""

    mod_basis: int = 113
    hidden_dim: int = 100
    batch_size: int = 128
    learning_rate: float = 3e-3
    weight_decay: float = 0.5
    epochs: int = 2000
    grad_accum_steps: int = 1
    checkpoint_dir: str = "checkpoints"
    checkpoint_every: int = 100
    device: str = "cuda" if th.cuda.is_available() else "cpu"
    compile: bool = True
    seed: int = 0


class BilinearModularModel(nn.Module):
    """Bilinear model for modular arithmetic.

    Instead of using a 2-layer MLP, we use a bilinear layer to learn modular addition.
    The network takes two one-hot encoded inputs and outputs logits for the result.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, use_output_projection: bool = True):
        super().__init__()
        self.input_dim = input_dim  # type: ignore[misc]
        self.hidden_dim = hidden_dim  # type: ignore[misc]
        self.output_dim = output_dim  # type: ignore[misc]
        self.use_output_projection = use_output_projection

        # Bilinear layer: combines two inputs via learned interaction
        self.bilinear = nn.Bilinear(input_dim, input_dim, hidden_dim, bias=True)

        # Optional output projection
        if use_output_projection:
            self.output = nn.Linear(hidden_dim, output_dim)
        else:
            self.output = None

    def forward(self, a: th.Tensor, b: th.Tensor) -> th.Tensor:
        """Forward pass.

        Args:
            a: First input, shape (batch_size, input_dim)
            b: Second input, shape (batch_size, input_dim)

        Returns:
            Logits for the output, shape (batch_size, output_dim)
        """
        # Bilinear interaction between inputs
        hidden = self.bilinear(a, b)

        # Project to output if using output projection
        logits = self.output(hidden) if self.use_output_projection else hidden

        return logits

    def get_interaction_matrices(self) -> th.Tensor:
        """Extract the bilinear interaction weight matrices.

        Returns:
            Tensor of shape (hidden_dim, input_dim, input_dim) representing
            the learned interaction matrices.
        """
        # The bilinear layer weight has shape (hidden_dim, input_dim, input_dim)
        return self.bilinear.weight.data


def save_checkpoint(
    model: nn.Module,
    optimizer: th.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_path: Path,
    config: TrainingConfig,
) -> None:
    """Save training checkpoint."""
    checkpoint_file = checkpoint_path / f"checkpoint_epoch_{epoch}.pt"
    th.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "config": vars(config),
        },
        checkpoint_file,
    )
    logger.info(f"Saved checkpoint to {checkpoint_file}")


def load_checkpoint(checkpoint_file: str | Path, model: nn.Module, optimizer: th.optim.Optimizer) -> int:
    """Load training checkpoint.

    Returns:
        Starting epoch number
    """
    checkpoint = th.load(checkpoint_file, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    logger.info(f"Loaded checkpoint from {checkpoint_file}, resuming from epoch {epoch}")
    return epoch


def train_epoch(
    model: nn.Module,
    optimizer: th.optim.Optimizer,
    criterion: nn.Module,
    train_loader: Any,  # TODO: Replace with actual dataloader type
    device: str,
    grad_accum_steps: int,
    tracker: trackio.Run,
    epoch: int,
) -> float:
    """Train for one epoch.

    Args:
        model: The model to train
        optimizer: Optimizer
        criterion: Loss function
        train_loader: Training data loader (TODO: implement in parallel)
        device: Device to train on
        grad_accum_steps: Number of gradient accumulation steps
        tracker: Trackio tracker for logging
        epoch: Current epoch number

    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # TODO: Once dataset is ready, replace this with actual data loading
    # For now, this is a placeholder structure
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)

    optimizer.zero_grad()

    for batch_idx, (a, b, targets) in enumerate(pbar):
        # Move to device
        a = a.to(device)
        b = b.to(device)
        targets = targets.to(device)

        # Forward pass
        logits = model(a, b)
        loss = criterion(logits, targets)

        # Scale loss for gradient accumulation
        loss = loss / grad_accum_steps

        # Backward pass
        loss.backward()

        # Update weights every grad_accum_steps
        if (batch_idx + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({"loss": loss.item() * grad_accum_steps})

        # Log to trackio
        tracker.log(
            {
                "train/batch_loss": loss.item() * grad_accum_steps,
                "train/step": epoch * len(train_loader) + batch_idx,
            }
        )

    return total_loss / num_batches


def validate(
    model: nn.Module,
    criterion: nn.Module,
    val_loader: Any,  # TODO: Replace with actual dataloader type
    device: str,
) -> tuple[float, float]:
    """Validate the model.

    Args:
        model: The model to validate
        criterion: Loss function
        val_loader: Validation data loader (TODO: implement in parallel)
        device: Device to validate on

    Returns:
        Tuple of (average validation loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # TODO: Once dataset is ready, replace this with actual data loading
    with th.no_grad():
        for a, b, targets in tqdm(val_loader, desc="Validating", leave=False):
            # Move to device
            a = a.to(device)
            b = b.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(a, b)
            loss = criterion(logits, targets)

            # Calculate accuracy
            predictions = th.argmax(logits, dim=1)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    return avg_loss, accuracy


@arguably.command
def train(
    *,
    mod_basis: int = 113,
    hidden_dim: int = 100,
    batch_size: int = 128,
    learning_rate: float = 3e-3,
    weight_decay: float = 0.5,
    epochs: int = 2000,
    grad_accum_steps: int = 1,
    checkpoint_dir: str = "checkpoints",
    checkpoint_every: int = 100,
    device: str | None = None,
    compile: bool = True,
    seed: int = 0,
    resume_from: str | None = None,
) -> None:
    """Train a bilinear layer on modular arithmetic.

    Args:
        mod_basis: The modulus for arithmetic (default: 113)
        hidden_dim: Hidden dimension size (default: 100)
        batch_size: Batch size for training (default: 128)
        learning_rate: Learning rate (default: 3e-3)
        weight_decay: Weight decay for AdamW (default: 0.5)
        epochs: Number of training epochs (default: 2000)
        grad_accum_steps: Gradient accumulation steps (default: 1)
        checkpoint_dir: Directory to save checkpoints (default: "checkpoints")
        checkpoint_every: Save checkpoint every N epochs (default: 100)
        device: Device to train on (default: auto-detect cuda/cpu)
        compile: Whether to use torch.compile (default: True)
        seed: Random seed (default: 42)
        resume_from: Path to checkpoint to resume from (default: None)
    """
    # Set up config
    if device is None:
        device = "cuda" if th.cuda.is_available() else "cpu"

    config = TrainingConfig(
        mod_basis=mod_basis,
        hidden_dim=hidden_dim,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=epochs,
        grad_accum_steps=grad_accum_steps,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=checkpoint_every,
        device=device,
        compile=compile,
        seed=seed,
    )

    logger.info(f"Training configuration: {config}")

    # Set random seed
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)

    # Set up checkpoint directory
    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Checkpoint directory: {checkpoint_path.absolute()}")

    # Initialize model
    input_dim = mod_basis
    output_dim = mod_basis
    model: nn.Module = BilinearModularModel(input_dim, hidden_dim, output_dim)
    model = model.to(device)

    # Compile model for speed
    if compile:
        logger.info("Compiling model with torch.compile...")
        model = th.compile(model)  # type: ignore[misc]

    # Initialize optimizer
    optimizer = th.optim.AdamW(
        model.parameters(),  # type: ignore[misc]
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Loss function
    nn.CrossEntropyLoss()

    # Load checkpoint if resuming
    start_epoch = 0
    if resume_from is not None:
        start_epoch = load_checkpoint(resume_from, model, optimizer)

    # Initialize tracker
    tracker = trackio.init(project="bilinear-modular-arithmetic")
    tracker.log({"config": config})

    # TODO: Set up data loaders
    # This will be implemented by the parallel agent
    # Expected interface:
    # train_loader, val_loader = load_data(mod_basis, batch_size)
    logger.warning("TODO: Data loading not implemented yet - waiting on parallel agent")
    logger.warning("Expected interface: train_loader, val_loader = load_data(mod_basis, batch_size)")

    # Placeholder for data loaders

    # Training loop
    logger.info(f"Starting training on device: {device}")

    for epoch in range(start_epoch, epochs):
        # TODO: Uncomment once data is ready
        # train_loss = train_epoch(
        #     model, optimizer, criterion, train_loader, device, grad_accum_steps, tracker, epoch
        # )

        # TODO: Uncomment once data is ready
        # val_loss, val_accuracy = validate(model, criterion, val_loader, device)

        # logger.info(
        #     f"Epoch {epoch}/{epochs} - "
        #     f"Train Loss: {train_loss:.4f} - "
        #     f"Val Loss: {val_loss:.4f} - "
        #     f"Val Accuracy: {val_accuracy:.4f}"
        # )

        # tracker.log(
        #     {
        #         "train/epoch_loss": train_loss,
        #         "val/loss": val_loss,
        #         "val/accuracy": val_accuracy,
        #         "epoch": epoch,
        #     }
        # )

        # Save checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            # TODO: Uncomment once training is ready
            # save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path, config)
            pass

    logger.info("Training complete!")
    tracker.finish()


if __name__ == "__main__":
    arguably.run()
