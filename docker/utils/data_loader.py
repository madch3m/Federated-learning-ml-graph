"""
Utility functions for loading datasets based on model selection.
"""
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from typing import Tuple


def get_dataset_for_model(model_name: str, data_dir: str = "./data") -> Tuple[Dataset, Dataset]:
    """
    Load the appropriate dataset based on the model name.
    
    Args:
        model_name: Name of the model ("cnn", "cifar10_cnn", "resnet")
        data_dir: Directory to store/load datasets
    
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    model_name = model_name.lower().strip()
    
    if model_name == "cifar10_cnn" or model_name == "cifar10cnn":
        return load_cifar10(data_dir)
    else:
        # Default to MNIST for cnn and resnet (unless resnet is configured for CIFAR10)
        return load_mnist(data_dir)


def load_mnist(data_dir: str = "./data") -> Tuple[Dataset, Dataset]:
    """Load MNIST dataset."""
    transform_train = transforms.Compose([transforms.ToTensor()])
    transform_test = transforms.Compose([transforms.ToTensor()])
    
    train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform_train)
    test = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform_test)
    
    return train, test


def load_cifar10(data_dir: str = "./data") -> Tuple[Dataset, Dataset]:
    """Load CIFAR10 dataset."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    
    return train, test

