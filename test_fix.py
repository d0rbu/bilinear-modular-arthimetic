"""Test the fixes."""

from pathlib import Path

import torch as th
import torch.nn as nn

from src.bilinear_modular.core.dataset import ModularArithmeticDataset
from src.bilinear_modular.core.train import BilinearModularModel

# Load dataset
mod_basis = 113
data_dir = Path(f"data/{mod_basis}")
dataset = ModularArithmeticDataset(
    mod_basis=mod_basis,
    data_dir=data_dir,
    train_split=0.8,
    one_hot=True,
    seed=0,
    batch_size=128,
)

# Initialize model with new settings
model = BilinearModularModel(mod_basis, 100, mod_basis, use_output_projection=False)

print("=== Model with bias ===")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}")

# Test forward pass
inputs, targets = next(iter(dataset.train()))
a = inputs[:, :mod_basis]
b = inputs[:, mod_basis:]

logits = model(a, b)
print(f"\nLogits range: [{logits.min().item():.6f}, {logits.max().item():.6f}]")
print(f"Logits std: {logits.std().item():.6f}")

# Test training with new hyperparameters
criterion = nn.CrossEntropyLoss()
optimizer = th.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

print("\n=== Training with new hyperparameters ===")
model.train()
for step in range(10):
    inputs, targets = next(iter(dataset.train()))
    a = inputs[:, :mod_basis]
    b = inputs[:, mod_basis:]

    logits = model(a, b)
    loss = criterion(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    predictions = th.argmax(logits, dim=1)
    targets_class = th.argmax(targets, dim=1)
    acc = (predictions == targets_class).float().mean().item()

    if step == 0:
        # Check gradient magnitude
        grad_std = model.bilinear.weight.grad.std().item()
        print(f"Initial gradient std: {grad_std:.6f}")

    print(f"Step {step}: loss={loss.item():.4f}, acc={acc:.4f}")

print("\nExpected: Loss should decrease and accuracy should improve!")
