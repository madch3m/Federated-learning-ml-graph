#!/usr/bin/env python3
"""
Standalone orchestration script for federated learning.
Runs federated learning training locally and saves results with evaluation and visualization.
"""
import torch
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, random_split
from pathlib import Path

from model import get_model, get_model_info
from utils.plotter import plot_history
from utils.experiments import create_experiment_dir
from utils.dirichlet_partition import dirichlet_partition
from utils.data_loader import get_dataset_for_model
from client.local_server import FederatedClient
from server.model_server import FederatedServer
import os


@dataclass
class HParams:
    num_clients: int = 20
    sample_clients: float = 0  # If 0 or <= 0, use all clients
    local_epochs: int = 2
    local_batch_size: int = 64
    rounds: int = 25
    lr: float = 0.01
    momentum: float = 0.0
    weight_decay: float = 0.0
    optimizer: str = "sgd"  # "sgd" or "adam"
    iid: bool = False
    dirichlet_alpha: float = 0.5
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_name: str = "cnn"  # Model to use: "cnn", "cifar10_cnn", or "resnet"


hp = HParams()
# Override model_name from environment variable if set
if os.getenv("MODEL_NAME"):
    hp.model_name = os.getenv("MODEL_NAME").lower().strip()
random.seed(hp.seed)
torch.manual_seed(hp.seed)


def load_data() -> Tuple[List[Subset], Dataset]:
    """Load and partition the dataset for federated learning based on model selection."""
    # Load appropriate dataset based on model
    dataset_name = "CIFAR10" if "cifar10" in hp.model_name.lower() else "MNIST"
    print(f"Loading {dataset_name} dataset for model '{hp.model_name}'...")
    
    train, test = get_dataset_for_model(hp.model_name, data_dir="./data")
    print(f"{dataset_name} dataset loaded: {len(train)} training samples, {len(test)} test samples")

    if hp.iid:
        # IID partitioning: random split
        sizes = [len(train) // hp.num_clients] * hp.num_clients
        sizes[-1] += len(train) - sum(sizes)
        shards = random_split(train, sizes, generator=torch.Generator().manual_seed(hp.seed))
        clients = [Subset(train, s.indices) for s in shards]
        return clients, test
    else:
        # Non-IID partitioning: Dirichlet distribution
        clients = dirichlet_partition(
            train=train,
            num_clients=hp.num_clients,
            alpha=hp.dirichlet_alpha,
            seed=hp.seed,
            ensure_min_size=True,
        )
        return clients, test


def create_clients(client_datasets: List[Subset]) -> List[FederatedClient]:
    """Create FederatedClient instances from datasets."""
    clients = []
    for client_id, dataset in enumerate(client_datasets):
        client = FederatedClient(
            client_id=client_id,
            dataset=dataset,
            local_epochs=hp.local_epochs,
            batch_size=hp.local_batch_size,
            lr=hp.lr,
            momentum=hp.momentum,
            weight_decay=hp.weight_decay,
            optimizer=hp.optimizer,
            device=hp.device,
        )
        clients.append(client)
    return clients


def orchestrate() -> Tuple[Dict[str, List[float]], FederatedServer, Dataset]:
    """Main federated learning orchestration loop."""
    # Load and partition data
    client_datasets, testset = load_data()
    
    # Create clients and server
    clients = create_clients(client_datasets)
    
    # Load the selected model
    try:
        model = get_model(hp.model_name)
        model_info = get_model_info(hp.model_name)
        print(f"Using model: {model_info.get('name', hp.model_name)} - {model_info.get('description', '')}")
    except ValueError as e:
        print(f"Warning: Failed to load model '{hp.model_name}': {e}")
        print("Falling back to default CNN model")
        model = get_model("cnn")
        hp.model_name = "cnn"
    
    server = FederatedServer(model=model, device=hp.device)
    
    # Initialize history tracking
    history: Dict[str, List[float]] = {"round": [], "acc": [], "loss": []}

    # Evaluate initial model (round 0)
    acc0, loss0 = server.evaluate(testset)
    history["round"].append(0)
    history["acc"].append(acc0)
    history["loss"].append(loss0)

    print(
        f"Simulated FedAvg with {hp.num_clients} clients, {hp.rounds} rounds, "
        f"{hp.local_epochs} local epochs (iid = {hp.iid})"
    )
    print(f"[Round 00] Test Acc: {acc0*100:5.2f}% | Test Loss: {loss0:.4f}")

    # Federated learning rounds
    for rnd in range(1, hp.rounds + 1):
        # Sample clients for this round
        # If sample_clients is 0 or <= 0, use all clients
        if hp.sample_clients <= 0:
            num_selected = hp.num_clients
        else:
            num_selected = max(1, int(hp.sample_clients * hp.num_clients))
        selected_indices = random.sample(range(hp.num_clients), min(num_selected, hp.num_clients))
        selected_clients = [clients[i] for i in selected_indices]

        # Collect updates from selected clients
        client_updates = []
        for client in selected_clients:
            state_dict, num_samples = client.train(server.get_global_model())
            client_updates.append((state_dict, num_samples))

        # Aggregate updates on server
        server.aggregate(client_updates)

        # Evaluate global model
        acc, loss = server.evaluate(testset)
        history["round"].append(rnd)
        history["acc"].append(acc)
        history["loss"].append(loss)

        print(f"[Round {rnd:02d}] Test Acc: {acc*100:5.2f}% | Test Loss: {loss:.4f}")

    return history, server, testset


if __name__ == "__main__":
    # Create experiment directory
    exp_dir: Path = create_experiment_dir(asdict(hp), exp_root="experiments")
    
    print("="*60)
    print(f"Starting Federated Learning Orchestration")
    print(f"Experiment directory: {exp_dir}")
    print("="*60)
    
    # Run orchestration
    history, server, testset = orchestrate()
    
    print("\n" + "="*60)
    print("Training completed! Saving model and generating visualization...")
    print("="*60)
    
    # Save the model
    model_path = exp_dir / "mnist_cnn.pt"
    server.save_model(str(model_path))
    print(f"Model saved to: {model_path}")
    
    # Run final evaluation on saved model
    print("\nRunning final evaluation on saved model...")
    final_acc, final_loss = server.evaluate(testset)
    
    print("="*60)
    print("FINAL MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"Final Test Accuracy: {final_acc*100:5.2f}%")
    print(f"Final Test Loss: {final_loss:.4f}")
    print(f"Total Rounds: {len(history['round'])}")
    
    # Display summary of all rounds
    if history["round"]:
        print("\nTraining Summary:")
        print("-" * 60)
        for rnd, acc, loss in zip(history["round"], history["acc"], history["loss"]):
            print(f"Round {rnd:2d}: Acc={acc*100:5.2f}% | Loss={loss:.4f}")
    
    print("="*60)
    
    # Generate and save visualization
    print("\nGenerating visualization...")
    try:
        plot_history(history, exp_dir=exp_dir, show=False)  # show=False for Docker/headless
        print(f"Visualization saved to: {exp_dir / 'metrics.png'}")
        print(f"Metrics CSV saved to: {exp_dir / 'metrics.csv'}")
    except Exception as e:
        print(f"Warning: Failed to generate visualization: {e}")
        print("Metrics CSV was still saved successfully.")
    
    print("="*60)
    print(f"[OK] Experiment completed successfully!")
    print(f"All outputs saved to: {exp_dir}")
    print(f"  - Model: {model_path}")
    print(f"  - Metrics CSV: {exp_dir / 'metrics.csv'}")
    print(f"  - Visualization: {exp_dir / 'metrics.png'}")
    print("="*60)

