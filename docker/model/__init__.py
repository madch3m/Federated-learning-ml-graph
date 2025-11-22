"""
Model factory for loading different model architectures.
"""
import torch.nn as nn
from typing import Optional

from .cnn import CNN, CIFAR10CNN
from .resnet import ResNet, resNetBlock


def get_model(model_name: str = "cnn", **kwargs) -> nn.Module:
    """
    Factory function to get a model by name.
    
    Args:
        model_name: Name of the model to load. Options:
            - "cnn": Simple CNN for MNIST (default)
            - "cifar10_cnn": CNN for CIFAR10
            - "resnet": ResNet model
        **kwargs: Additional arguments for model initialization
    
    Returns:
        Initialized model instance
    
    Raises:
        ValueError: If model_name is not recognized
    """
    model_name = model_name.lower().strip()
    
    if model_name == "cnn":
        return CNN()
    elif model_name == "cifar10_cnn" or model_name == "cifar10cnn":
        num_classes = kwargs.get("num_classes", 10)
        return CIFAR10CNN(num_classes=num_classes)
    elif model_name == "resnet":
        num_blocks = kwargs.get("num_blocks", [2, 2, 2, 2])
        num_classes = kwargs.get("num_classes", 10)
        in_channels = kwargs.get("in_channels", 1)  # 1 for MNIST, 3 for CIFAR10
        return ResNet(resNetBlock, num_blocks, num_classes=num_classes, in_channels=in_channels)
    else:
        raise ValueError(
            f"Unknown model name: {model_name}. "
            f"Available models: 'cnn', 'cifar10_cnn', 'resnet'"
        )


def get_model_info(model_name: str) -> dict:
    """
    Get information about a model.
    
    Args:
        model_name: Name of the model
    
    Returns:
        Dictionary with model information
    """
    model_name = model_name.lower().strip()
    
    info = {
        "cnn": {
            "name": "CNN",
            "description": "Simple CNN for MNIST",
            "input_channels": 1,
            "input_size": (28, 28),
            "num_classes": 10
        },
        "cifar10_cnn": {
            "name": "CIFAR10CNN",
            "description": "CNN for CIFAR10",
            "input_channels": 3,
            "input_size": (32, 32),
            "num_classes": 10
        },
        "resnet": {
            "name": "ResNet",
            "description": "ResNet architecture (configurable)",
            "input_channels": "configurable (1 or 3)",
            "input_size": "configurable",
            "num_classes": "configurable"
        }
    }
    
    return info.get(model_name, {"name": "Unknown", "description": "Model not found"})

