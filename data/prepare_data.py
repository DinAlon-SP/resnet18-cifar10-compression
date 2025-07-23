import os
from collections import Counter

import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit



# ========== MAIN FUNCTION ==========

def get_cifar10_datasets(seed=42, from_disk=False, path="data/processed"):
    if from_disk and _check_cache_exists(path):
        return _load_from_disk(path)

    # Transforms
    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2470, 0.2435, 0.2616]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Load dataset
    train_val_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=train_transform)
    labels = np.array(train_val_set.targets)

    # Stratified split: 80% train / 20% val
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))
    train_set = Subset(train_val_set, train_idx)
    val_set = Subset(train_val_set, val_idx)

    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=test_transform)

    class_names = train_val_set.classes
    full_labels = labels

    return train_set, val_set, test_set, class_names, full_labels

# ========== DISK CACHE HELPERS ==========

def save_to_disk(train_set, val_set, test_set, class_names, full_labels, path="data/processed"):
    os.makedirs(path, exist_ok=True)
    torch.save(train_set, os.path.join(path, "train_set.pt"))
    torch.save(val_set,   os.path.join(path, "val_set.pt"))
    torch.save(test_set,  os.path.join(path, "test_set.pt"))
    torch.save(class_names, os.path.join(path, "class_names.pt"))
    torch.save(full_labels, os.path.join(path, "full_labels.pt"))

def _load_from_disk(path):
    train_set = torch.load(os.path.join(path, "train_set.pt"))
    val_set   = torch.load(os.path.join(path, "val_set.pt"))
    test_set  = torch.load(os.path.join(path, "test_set.pt"))
    class_names = torch.load(os.path.join(path, "class_names.pt"))
    full_labels = torch.load(os.path.join(path, "full_labels.pt"))
    return train_set, val_set, test_set, class_names, full_labels

def _check_cache_exists(path):
    files = ["train_set.pt", "val_set.pt", "test_set.pt", "class_names.pt", "full_labels.pt"]
    return all(os.path.exists(os.path.join(path, f)) for f in files)

# ========== STATS ==========

def get_subset_class_counts(subset, full_labels):
    subset_labels = [full_labels[i] for i in subset.indices]
    return Counter(subset_labels)

def print_dataset_summary(train_set, val_set, test_set, class_names, full_labels):
    train_counts = get_subset_class_counts(train_set, full_labels)
    val_counts   = get_subset_class_counts(val_set, full_labels)
    test_counts  = Counter(test_set.targets)

    print(f"\n Dataset Sizes:")
    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    print("\n Class distribution:")
    print(f"{'Class':<15}{'Train Count':<15}{'Val Count':<15}{'Test Count'}")
    print("-" * 55)
    for i, class_name in enumerate(class_names):
        print(f"{class_name:<15}{train_counts[i]:<15}{val_counts[i]:<15}{test_counts[i]}")

# ========== EXAMPLE USAGE ==========

if __name__ == "__main__":
    # Set from_disk=True to load previously saved datasets
    train_set, val_set, test_set, class_names, full_labels = get_cifar10_datasets(seed=42, from_disk=False)

    print_dataset_summary(train_set, val_set, test_set, class_names, full_labels)

    # Save to disk for fast reuse
    save_to_disk(train_set, val_set, test_set, class_names, full_labels)
    print("\n Datasets saved to disk.")

