# main.py
import torch, random
import torch.nn as nn
import torch.optim as optim

from copy import deepcopy
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from tqdm import tqdm
from pathlib import Path

from model.cnn import CNN
from utils.plotter import plot_history
from utils.experiments import create_experiment_dir
from utils.dirichlet_partition import dirichlet_partition


@dataclass
class HParams:
    num_clients: int = 20
    sample_clients: float = 0.25
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
    transform = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    if hp.iid:
        sizes = [len(train) // hp.num_clients] * hp.num_clients
        sizes[-1] += len(train) - sum(sizes)
        shards = random_split(train, sizes, generator=torch.Generator().manual_seed(hp.seed))
        clients = [Subset(train, s.indices) for s in shards]
        return clients, test
    else:
        clients = dirichlet_partition(
            train=train,
            num_clients=hp.num_clients,
            alpha=hp.dirichlet_alpha,
            seed=hp.seed,
            ensure_min_size=True,
        )
        return clients, test


def client_update(global_model: nn.Module, dataset: Subset) -> Tuple[Dict[str, torch.Tensor], int]:
    model = deepcopy(global_model).to(hp.device)
    model.train()
    loader = DataLoader(dataset, batch_size=hp.local_batch_size, shuffle=True, drop_last=False)
    criterion = nn.CrossEntropyLoss()

    if hp.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    else:
        optimizer = optim.SGD(
            model.parameters(), lr=hp.lr, momentum=hp.momentum, weight_decay=hp.weight_decay
        )

    for epoch in range(hp.local_epochs):
        for x, y in tqdm(loader, desc=f"Local epoch {epoch+1}/{hp.local_epochs}", leave=False):
            x, y = x.to(hp.device), y.to(hp.device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    return deepcopy(model.state_dict()), len(dataset)

@torch.no_grad()
def fedavg(global_model: nn.Module, client_states: List[Tuple[Dict[str, torch.Tensor], int]]) -> None:
    total_samples = sum(n for _, n in client_states)
    avg_state: Dict[str, torch.Tensor] = {}

    for k, v in global_model.state_dict().items():
        if v.dtype.is_floating_point or v.dtype.is_complex:
            avg_state[k] = torch.zeros_like(v, device=hp.device, dtype=v.dtype)
        else:
            avg_state[k] = v.clone().to(hp.device)

    for state_dict, n in client_states:
        w = n / total_samples
        for k in avg_state.keys():
            t = state_dict[k].to(hp.device)
            if t.dtype.is_floating_point or t.dtype.is_complex:
                avg_state[k] += t * w
            else:
                avg_state[k] = t

    global_model.load_state_dict(avg_state)

@torch.no_grad()
def evaluate(model: nn.Module, testset: Dataset) -> Tuple[float, float]:
    model.eval().to(hp.device)
    loader = DataLoader(testset, batch_size=512, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    correct, total, total_loss = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(hp.device), y.to(hp.device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total, total_loss / total


def orchestrate() -> Tuple[Dict[str, List[float]], nn.Module]:
    clients, testset = load_data()
    global_model = CNN().to(hp.device)
    history: Dict[str, List[float]] = {"round": [], "acc": [], "loss": []}

    # round 0 baseline
    acc0, loss0 = evaluate(global_model, testset)
    history["round"].append(0)
    history["acc"].append(acc0)
    history["loss"].append(loss0)

    print(
        f"Simulated FedAvg with {hp.num_clients} clients, {hp.rounds} rounds, "
        f"{hp.local_epochs} local epochs (iid = {hp.iid})"
    )

    for rnd in range(1, hp.rounds + 1):
        m = max(1, int(hp.sample_clients * hp.num_clients))
        selected = random.sample(range(hp.num_clients), m)

        client_states: List[Tuple[Dict[str, torch.Tensor], int]] = []
        for cid in selected:
            state, n_samples = client_update(global_model, clients[cid])
            client_states.append((state, n_samples))

        fedavg(global_model, client_states)

        acc, loss = evaluate(global_model, testset)
        history["round"].append(rnd)
        history["acc"].append(acc)
        history["loss"].append(loss)

        try:
            tqdm.write(f"[Round {rnd:02d}] Test Acc: {acc*100:5.2f}% | Test Loss: {loss:.4f}")
        except NameError:
            print(f"[Round {rnd:02d}] Test Acc: {acc*100:5.2f}% | Test Loss: {loss:.4f}")

    return history, global_model


if __name__ == "__main__":
    exp_dir: Path = create_experiment_dir(asdict(hp))
    history, model = orchestrate()
    torch.save(model.state_dict(), exp_dir / "mnist_cnn.pt")
    plot_history(history, exp_dir=exp_dir, show=True)
    print(f"[OK] Experiment saved to: {exp_dir}")