#!/usr/bin/env python3
"""
Client runner script for Docker deployment.
Reads configuration from environment variables.
"""
import os
import sys
import torch
import random
import logging
from torchvision import datasets, transforms
from torch.utils.data import Subset

from middleware.network_client import create_network_client
from utils.dirichlet_partition import dirichlet_partition
from utils.data_loader import get_dataset_for_model
from client.local_server import FederatedClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_client_data(client_id: int, num_clients: int, alpha: float = 0.5, seed: int = 42):
    """
    Load and partition data for a specific client based on model selection.
    
    Args:
        client_id: ID of this client
        num_clients: Total number of clients in the federation
        alpha: Dirichlet alpha parameter for non-IID partitioning
        seed: Random seed for reproducibility
        
    Returns:
        Dataset subset for this client
    """
    # Get model name from environment variable
    model_name = os.getenv("MODEL_NAME", "cnn").lower().strip()
    dataset_name = "CIFAR10" if "cifar10" in model_name else "MNIST"
    
    logger.info(f"Loading {dataset_name} data for client {client_id}/{num_clients} (model: {model_name})")
    
    # Load appropriate dataset
    train, _ = get_dataset_for_model(model_name, data_dir="./data")
    
    # Partition the data using Dirichlet distribution
    try:
        clients = dirichlet_partition(
            train=train,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
            ensure_min_size=True,
        )
        
        if client_id >= len(clients):
            logger.error(f"Client ID {client_id} exceeds number of clients {len(clients)}")
            sys.exit(1)
        
        client_dataset = clients[client_id]
        logger.info(f"Client {client_id} has {len(client_dataset)} samples from {dataset_name}")
        
        return client_dataset
        
    except ImportError:
        logger.warning("Dirichlet partition not available, using random split")
        # Fallback to random split
        from torch.utils.data import random_split
        sizes = [len(train) // num_clients] * num_clients
        sizes[-1] += len(train) - sum(sizes)
        shards = random_split(train, sizes, generator=torch.Generator().manual_seed(seed))
        return Subset(train, shards[client_id].indices)


def main():
    """Main entry point for the client."""
    
    # Read configuration from environment variables
    server_url = os.getenv("SERVER_URL", "http://server:8000")
    client_id = int(os.getenv("CLIENT_ID", "0"))
    num_clients = int(os.getenv("NUM_CLIENTS", "5"))
    device = os.getenv("DEVICE", "cpu")
    local_epochs = int(os.getenv("LOCAL_EPOCHS", "2"))
    batch_size = int(os.getenv("BATCH_SIZE", "64"))
    lr = float(os.getenv("LEARNING_RATE", "0.01"))
    num_rounds = int(os.getenv("NUM_ROUNDS", "25"))
    alpha = float(os.getenv("DIRICHLET_ALPHA", "0.5"))
    seed = int(os.getenv("SEED", "42"))
    start_delay = float(os.getenv("START_DELAY", "0"))
    
    logger.info("="*60)
    logger.info(f"Starting Federated Learning Client")
    logger.info("="*60)
    logger.info(f"Client ID: {client_id}")
    logger.info(f"Server URL: {server_url}")
    logger.info(f"Device: {device}")
    logger.info(f"Local epochs: {local_epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {lr}")
    logger.info(f"Number of rounds: {num_rounds}")
    logger.info(f"Dirichlet alpha: {alpha}")
    logger.info(f"Start delay: {start_delay}s")
    logger.info("="*60)
    
    # Set random seeds
    random.seed(seed + client_id)
    torch.manual_seed(seed + client_id)
    
    # Load client's data
    try:
        dataset = load_client_data(
            client_id=client_id,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        sys.exit(1)
    
    # Create network-tolerant client
    try:
        client = create_network_client(
            client_id=client_id,
            dataset=dataset,
            server_url=server_url,
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )
    except Exception as e:
        logger.error(f"Failed to create client: {e}")
        sys.exit(1)
    
    # Participate in federated learning
    try:
        logger.info(f"Client {client_id} starting participation in {num_rounds} rounds")
        client.participate(num_rounds=num_rounds, start_delay=start_delay)
        logger.info(f"Client {client_id} completed all rounds successfully")
    except KeyboardInterrupt:
        logger.info(f"Client {client_id} interrupted by user")
    except Exception as e:
        logger.error(f"Client {client_id} encountered an error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()