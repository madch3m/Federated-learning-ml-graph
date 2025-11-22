#!/usr/bin/env python3
"""
Test script for visualization logic using MNIST dataset.
Creates two small experiments and tests the visualization.
"""
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, random_split

from model import get_model
from utils.plotter import plot_history
from utils.experiments import create_experiment_dir, visualize_last_two_experiments
from utils.dirichlet_partition import dirichlet_partition
from client.local_server import FederatedClient
from server.model_server import FederatedServer


@dataclass
class TestHParams:
    num_clients: int = 5
    local_epochs: int = 1
    local_batch_size: int = 32
    rounds: int = 5  # Small number for testing
    lr: float = 0.01
    momentum: float = 0.0
    weight_decay: float = 0.0
    optimizer: str = "sgd"
    iid: bool = False
    dirichlet_alpha: float = 0.5
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_name: str = "cnn"


def run_small_experiment(hparams: TestHParams, exp_name: str) -> Path:
    """Run a small federated learning experiment and save metrics."""
    print(f"\n{'='*60}")
    print(f"Running Experiment: {exp_name}")
    print(f"{'='*60}")
    
    # Set random seeds
    random.seed(hparams.seed)
    torch.manual_seed(hparams.seed)
    
    # Load MNIST data
    print("Loading MNIST dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    print(f"Training samples: {len(train)}, Test samples: {len(test)}")
    
    # Partition data
    if hparams.iid:
        sizes = [len(train) // hparams.num_clients] * hparams.num_clients
        sizes[-1] += len(train) - sum(sizes)
        shards = random_split(train, sizes, generator=torch.Generator().manual_seed(hparams.seed))
        client_datasets = [Subset(train, s.indices) for s in shards]
    else:
        client_datasets = dirichlet_partition(
            train=train,
            num_clients=hparams.num_clients,
            alpha=hparams.dirichlet_alpha,
            seed=hparams.seed,
            ensure_min_size=True,
        )
    
    # Create clients
    print(f"Creating {hparams.num_clients} clients...")
    clients = []
    for client_id, dataset in enumerate(client_datasets):
        client = FederatedClient(
            client_id=client_id,
            dataset=dataset,
            local_epochs=hparams.local_epochs,
            batch_size=hparams.local_batch_size,
            lr=hparams.lr,
            momentum=hparams.momentum,
            weight_decay=hparams.weight_decay,
            optimizer=hparams.optimizer,
            device=hparams.device,
        )
        clients.append(client)
        print(f"  Client {client_id}: {len(dataset)} samples")
    
    # Create server
    print(f"Creating server with model: {hparams.model_name}")
    model = get_model(hparams.model_name)
    server = FederatedServer(model=model, device=hparams.device)
    
    # Create experiment directory
    hparams_dict = asdict(hparams)
    hparams_dict["model_name"] = hparams.model_name
    exp_dir = create_experiment_dir(hparams_dict, exp_root="experiments/test_vis", name=exp_name)
    print(f"Experiment directory: {exp_dir}")
    
    # Initialize history
    history: Dict[str, List[float]] = {"round": [], "acc": [], "loss": []}
    
    # Evaluate initial model (round 0)
    print("\nEvaluating initial model...")
    acc0, loss0 = server.evaluate(test)
    history["round"].append(0)
    history["acc"].append(acc0)
    history["loss"].append(loss0)
    print(f"Round 0: Acc={acc0*100:5.2f}% | Loss={loss0:.4f}")
    
    # Federated learning rounds
    print(f"\nRunning {hparams.rounds} federated learning rounds...")
    for rnd in range(1, hparams.rounds + 1):
        print(f"  Round {rnd}/{hparams.rounds}...", end=" ", flush=True)
        
        # Collect updates from all clients
        client_updates = []
        for client in clients:
            state_dict, num_samples = client.train(server.get_global_model())
            client_updates.append((state_dict, num_samples))
        
        # Aggregate updates
        server.aggregate(client_updates)
        
        # Evaluate global model
        acc, loss = server.evaluate(test)
        history["round"].append(rnd)
        history["acc"].append(acc)
        history["loss"].append(loss)
        
        print(f"Acc={acc*100:5.2f}% | Loss={loss:.4f}")
    
    # Save model
    model_path = exp_dir / "mnist_cnn.pt"
    server.save_model(str(model_path))
    print(f"\nModel saved to: {model_path}")
    
    # Generate and save visualization
    print("Generating metrics visualization...")
    plot_history(history, exp_dir=exp_dir, show=False)
    print(f"Metrics saved to: {exp_dir / 'metrics.csv'}")
    print(f"Visualization saved to: {exp_dir / 'metrics.png'}")
    
    print(f"\n✓ Experiment '{exp_name}' completed!")
    return exp_dir


def test_visualization():
    """Test the visualization logic with two small experiments."""
    print("="*60)
    print("Testing Visualization Logic with MNIST Dataset")
    print("="*60)
    
    # Create test experiments directory
    test_exp_dir = Path("experiments/test_vis")
    test_exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Experiment 1: Lower learning rate
    print("\n" + "="*60)
    print("EXPERIMENT 1: Lower Learning Rate")
    print("="*60)
    hp1 = TestHParams(
        lr=0.01,
        rounds=5,
        seed=42,
        model_name="cnn"
    )
    exp1_dir = run_small_experiment(hp1, "test_exp1_lr=0.01")
    
    # Experiment 2: Higher learning rate
    print("\n" + "="*60)
    print("EXPERIMENT 2: Higher Learning Rate")
    print("="*60)
    hp2 = TestHParams(
        lr=0.05,  # Different learning rate
        rounds=5,
        seed=42,
        model_name="cnn"
    )
    exp2_dir = run_small_experiment(hp2, "test_exp2_lr=0.05")
    
    # Test visualization
    print("\n" + "="*60)
    print("TESTING VISUALIZATION")
    print("="*60)
    
    try:
        result = visualize_last_two_experiments(
            exp_root="experiments/test_vis",
            output_path="experiments/test_vis/comparison_test.png"
        )
        
        if result and result.exists():
            print(f"\n✓ Visualization test PASSED!")
            print(f"  Comparison saved to: {result}")
            print(f"  File size: {result.stat().st_size / 1024:.1f} KB")
            
            # Verify the visualization file
            if result.stat().st_size > 0:
                print("  ✓ Visualization file is valid (non-empty)")
                return True
            else:
                print("  ✗ Visualization file is empty")
                return False
        else:
            print("\n✗ Visualization test FAILED: No output file created")
            return False
            
    except Exception as e:
        print(f"\n✗ Visualization test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    try:
        success = test_visualization()
        
        print("\n" + "="*60)
        if success:
            print("✓ ALL TESTS PASSED!")
            print("="*60)
            print("\nTest results:")
            print("  - Two experiments created with MNIST dataset")
            print("  - Metrics saved to CSV files")
            print("  - Individual visualizations generated")
            print("  - Comparison visualization created successfully")
            print(f"\nView the comparison:")
            print(f"  open experiments/test_vis/comparison_test.png")
            return 0
        else:
            print("✗ TESTS FAILED")
            print("="*60)
            return 1
            
    except Exception as e:
        print(f"\n✗ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

