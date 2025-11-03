import torch
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, random_split
from pathlib import Path

from model.cnn import CNN
from utils.plotter import plot_history
from utils.experiments import create_experiment_dir
from utils.dirichlet_partition import dirichlet_partition
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


hp = HParams()
random.seed(hp.seed)
torch.manual_seed(hp.seed)


def load_data() -> Tuple[List[Subset], Dataset]:
    """Load and partition the MNIST dataset for federated learning."""
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

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


def orchestrate() -> Tuple[Dict[str, List[float]], FederatedServer]:
    """Main federated learning orchestration loop."""
    # Load and partition data
    client_datasets, testset = load_data()
    
    # Create clients and server
    clients = create_clients(client_datasets)
    server = FederatedServer(model=CNN(), device=hp.device)
    
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

    return history, server


if __name__ == "__main__":
    # Create experiment directory
    # The create_experiment_dir function handles both "batch" and "local_batch_size" keys
    exp_dir: Path = create_experiment_dir(asdict(hp), exp_root="experiments")
    
    print("="*60)
    print(f"Starting Federated Learning Experiment")
    print(f"Experiment directory: {exp_dir}")
    print("="*60)
    
    history, server = orchestrate()
    
    # Save the model
    model_path = exp_dir / "mnist_cnn.pt"
    server.save_model(str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    # Generate visualization (show=False for Docker/headless environments)
    print("Generating visualization...")
    plot_history(history, exp_dir=exp_dir, show=False)
    
    print("="*60)
    print(f"[OK] Experiment completed successfully!")
    print(f"Results saved to: {exp_dir}")
    print(f"  - Model: {model_path}")
    print(f"  - Metrics CSV: {exp_dir / 'metrics.csv'}")
    print(f"  - Visualization: {exp_dir / 'metrics.png'}")
    print("="*60)