"""Debug script to understand the training issue."""

from pathlib import Path

import torch as th
import torch.nn as nn

from src.bilinear_modular.core.dataset import ModularArithmeticDataset, generate_dataset
from src.bilinear_modular.core.train import BilinearModularModel

# Generate small dataset if needed
mod_basis = 113
data_dir = Path(f"data/{mod_basis}")
if not (data_dir / "metadata.json").exists():
    print(f"Generating dataset for mod {mod_basis}...")
    generate_dataset(mod_basis)

# Load dataset
dataset = ModularArithmeticDataset(
    mod_basis=mod_basis,
    data_dir=data_dir,
    train_split=0.8,
    one_hot=True,
    seed=0,
    batch_size=128,
)

# Initialize model
input_dim = mod_basis
output_dim = mod_basis
hidden_dim = 100
model = BilinearModularModel(input_dim, hidden_dim, output_dim, use_output_projection=False)

# Check model parameters
print("\n=== Model Architecture ===")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")

# Check weight initialization
print("\n=== Weight Statistics ===")
for name, param in model.named_parameters():
    print(
        f"{name}: mean={param.mean().item():.6f}, std={param.std().item():.6f}, "
        f"min={param.min().item():.6f}, max={param.max().item():.6f}"
    )

# Get a batch and check the forward pass
print("\n=== Forward Pass Check ===")
inputs, targets = next(iter(dataset.train()))
mod_basis_check = inputs.shape[1] // 2
a = inputs[:, :mod_basis_check]
b = inputs[:, mod_basis_check:]

print(f"Input a shape: {a.shape}, range: [{a.min():.2f}, {a.max():.2f}]")
print(f"Input b shape: {b.shape}, range: [{b.min():.2f}, {b.max():.2f}]")
print(f"Targets shape: {targets.shape}")

# Forward pass
logits = model(a, b)
print(f"Output logits shape: {logits.shape}")
print(f"Logits stats: mean={logits.mean().item():.6f}, std={logits.std().item():.6f}")
print(f"Logits range: [{logits.min().item():.6f}, {logits.max().item():.6f}]")

# Check loss
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)
print(f"\nInitial loss: {loss.item():.6f}")

# Check what the model predicts
predictions = th.argmax(logits, dim=1)
targets_class = th.argmax(targets, dim=1)
print(f"\nPredictions (first 10): {predictions[:10].tolist()}")
print(f"Targets (first 10): {targets_class[:10].tolist()}")
print(f"Accuracy: {(predictions == targets_class).float().mean().item():.6f}")

# Random baseline
print(f"\nRandom baseline accuracy: {1 / mod_basis:.6f}")

# Check gradient flow
print("\n=== Gradient Check ===")
optimizer = th.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.5)
optimizer.zero_grad()
loss.backward()

for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name} grad: mean={param.grad.mean().item():.6f}, std={param.grad.std().item():.6f}")
    else:
        print(f"{name}: No gradient!")

# Try a few optimization steps
print("\n=== Training Steps ===")
model.train()
for step in range(5):
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

    print(f"Step {step}: loss={loss.item():.6f}, acc={acc:.6f}")
