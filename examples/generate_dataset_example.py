#!/usr/bin/env python
"""Example script for generating modular arithmetic datasets."""

from bilinear_modular import ModularArithmeticDataset, generate_dataset

# Generate dataset for mod 113 (as in the reference implementation)
print("=" * 60)
print("Generating dataset for mod 113")
print("=" * 60)
dataset = generate_dataset(mod_basis=113)

print("\nDataset Statistics:")
print(f"  Total samples: {len(dataset)}")
print(f"  Training samples: {dataset.train_size}")
print(f"  Validation samples: {dataset.val_size}")
print(f"  Input dimension (one-hot): {dataset.metadata['input_dim']}")
print(f"  Output dimension (one-hot): {dataset.metadata['output_dim']}")

# Example: Get a batch of training data
print("\n" + "=" * 60)
print("Getting a batch of training data")
print("=" * 60)
inputs, targets = dataset.get_train_batch(batch_size=5)
print(f"Inputs shape: {inputs.shape}")
print(f"Targets shape: {targets.shape}")
print(f"Inputs dtype: {inputs.dtype}")
print(f"Targets dtype: {targets.dtype}")

# Example: Verify a specific calculation
print("\n" + "=" * 60)
print("Example calculations")
print("=" * 60)
a_idx, b_idx = 42, 71
expected_c = (a_idx + b_idx) % 113

# Find this sample in the dataset
for i in range(len(dataset)):
    if dataset.a_values[i].item() == a_idx and dataset.b_values[i].item() == b_idx:
        actual_c = dataset.c_values[i].item()
        print(f"  {a_idx} + {b_idx} ≡ {actual_c} (mod 113)")
        print(f"  Expected: {expected_c}, Actual: {actual_c}, Match: {expected_c == actual_c}")
        break

# Example: Load an existing dataset
print("\n" + "=" * 60)
print("Reloading dataset from disk")
print("=" * 60)
dataset_reloaded = ModularArithmeticDataset(mod_basis=113)
print(f"  Successfully loaded {len(dataset_reloaded)} samples")

# Example: Iterator protocol for training
print("\n" + "=" * 60)
print("Using iterator for training loop")
print("=" * 60)
dataset.batch_size = 64
dataset.train()
for batch_idx, (inputs, _targets) in enumerate(dataset, 1):
    if batch_idx >= 3:  # Just show first 3 batches
        print(f"  Processed {batch_idx} batches, last batch shape: {inputs.shape}")
        break

print("\n✅ All examples completed successfully!")
