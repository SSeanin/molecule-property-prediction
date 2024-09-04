from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch_geometric.datasets import QM9

from .transforms import CompleteGraph, SetTarget


path = './data'
target = 0

transform = Compose([CompleteGraph(), SetTarget()])

# Load the QM9 dataset with the transforms defined
dataset = QM9(path, transform=transform, force_reload=False)

# Normalize targets per data sample to mean = 0 and std = 1.
mean = dataset.data.y.mean(dim=0, keepdim=True)
std = dataset.data.y.std(dim=0, keepdim=True)
dataset.data.y = (dataset.data.y - mean) / std
mean, std = mean[:, target].item(), std[:, target].item()

print(f"Total number of samples: {len(dataset)}.")

# Split datasets
test_dataset = dataset[:10000]
val_dataset = dataset[10000:20000]
train_dataset = dataset[20000:]

print(
    f"Created dataset splits with {len(train_dataset)} training, {len(val_dataset)} validation, {len(test_dataset)} test samples.")

# Create dataloaders with batch size = 32
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
