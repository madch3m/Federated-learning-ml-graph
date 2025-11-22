from typing import Dict, List
from pathlib import Path

import csv
import matplotlib.pyplot as plt

def plot_history(history: Dict[str, List[float]], exp_dir: str | Path | None = None, show: bool = True) -> None:
    rounds = history.get("round", [])
    acc = history.get("acc", [])
    loss = history.get("loss", [])

    # Saves CSV if exp_dir provided
    if exp_dir is not None:
        exp_dir = Path(exp_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        csv_path = exp_dir / "metrics.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["round", "test_accuracy", "test_loss"])
            for r, a, l in zip(rounds, acc, loss):
                w.writerow([r, a, l])

    # Plot (and optionally saves PNG alongside CSV)
    if rounds:
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(rounds, acc, marker='o')
        plt.title("Federated Learning - Test Accuracy")
        plt.xlabel("Round"); plt.ylabel("Accuracy")

        plt.subplot(1, 2, 2)
        plt.plot(rounds, loss, marker='o', color='red')
        plt.title("Federated Learning - Test Loss")
        plt.xlabel("Round"); plt.ylabel("Loss")

        plt.tight_layout()

        if exp_dir is not None:
            plt.savefig(Path(exp_dir) / "metrics.png", dpi=160, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close()
    else:
        print("History is empty—did training run?")
