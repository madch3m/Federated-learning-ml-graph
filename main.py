import sys
import os
from pathlib import Path

# Add docker directory to Python path to import modules
docker_dir = Path(__file__).parent / "docker"
if str(docker_dir) not in sys.path:
    sys.path.insert(0, str(docker_dir))

import torch
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, random_split

from model import get_model
from utils.plotter import plot_history
from utils.experiments import create_experiment_dir
from utils.dirichlet_partition import dirichlet_partition
from utils.data_loader import get_dataset_for_model
from client.local_server import FederatedClient
from server.model_server import FederatedServer


@dataclass
class HParams:
    num_clients: int = 20
    sample_clients: float = 0
    local_epochs: int = 2
    local_batch_size: int = 64
    rounds: int = 25
    lr: float = 0.01
    momentum: float = 0.0
    weight_decay: float = 0.0
    optimizer: str = "sgd"           # "sgd" or "adam"
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
    """Load and partition the dataset for federated learning."""
    # Load appropriate dataset based on model
    train, test = get_dataset_for_model(hp.model_name, data_dir="./data")
    
    # Partition data
    if hp.iid:
        # IID partition: random split
        sample_size = [len(train) // hp.num_clients] * hp.num_clients
        sample_size[-1] += len(train) - sum(sample_size)
        shards = random_split(train, sample_size, generator=torch.Generator().manual_seed(hp.seed))
        client_datasets = [Subset(train, s.indices) for s in shards]
    else:
        # Non-IID partition: Dirichlet distribution
        client_datasets = dirichlet_partition(
            train=train,
            num_clients=hp.num_clients,
            alpha=hp.dirichlet_alpha,
            seed=hp.seed,
            ensure_min_size=True,
        )
    
    return client_datasets, test


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


def orchestrate() -> Tuple[Dict[str, List[float]], FederatedServer]:
    """Main federated learning orchestration loop."""
    # Load and partition data
    client_datasets, testset = load_data()
    
    # Create model based on model_name
    model = get_model(hp.model_name)
    
    # Create clients and server
    clients = create_clients(client_datasets)
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
        num_selected = max(1, int(hp.sample_clients * hp.num_clients))
        selected_indices = random.sample(range(hp.num_clients), num_selected)
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

    return history, server


if __name__ == "__main__":
    exp_dir: Path = create_experiment_dir(asdict(hp))
    history, server = orchestrate()
    model_filename = f"{hp.model_name}_model.pt"
    server.save_model(str(exp_dir / model_filename))
    plot_history(history, exp_dir=exp_dir, show=True)
    print(f"[OK] Experiment saved to: {exp_dir}")