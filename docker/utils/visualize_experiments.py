"""
Visualize metrics from the last two experiments on a single page.
Finds experiment directories and creates a comparison visualization.
"""
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import matplotlib
from typing import List, Tuple, Optional
from datetime import datetime

# Use non-interactive backend for headless environments
matplotlib.use('Agg')


def find_experiment_directories(exp_root: str = "experiments") -> List[Tuple[Path, datetime]]:
    """
    Find all experiment directories that contain metrics.csv files.
    Returns list of (directory_path, modification_time) tuples, sorted by modification time.
    """
    exp_root = Path(exp_root)
    if not exp_root.exists():
        return []
    
    experiments = []
    
    # Search for directories containing metrics.csv
    for metrics_file in exp_root.rglob("metrics.csv"):
        exp_dir = metrics_file.parent
        # Get modification time of the directory or the metrics file
        mod_time = datetime.fromtimestamp(exp_dir.stat().st_mtime)
        experiments.append((exp_dir, mod_time))
    
    # Sort by modification time (most recent first)
    experiments.sort(key=lambda x: x[1], reverse=True)
    
    return experiments


def load_metrics(exp_dir: Path) -> Optional[dict]:
    """
    Load metrics from a metrics.csv file.
    Returns dict with 'round', 'acc', 'loss' lists, or None if file doesn't exist.
    """
    metrics_file = exp_dir / "metrics.csv"
    if not metrics_file.exists():
        return None
    
    rounds = []
    acc = []
    loss = []
    
    try:
        with metrics_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rounds.append(int(row["round"]))
                acc.append(float(row["test_accuracy"]))
                loss.append(float(row["test_loss"]))
        
        return {
            "round": rounds,
            "acc": acc,
            "loss": loss
        }
    except Exception as e:
        print(f"Error loading metrics from {metrics_file}: {e}")
        return None


def get_experiment_name(exp_dir: Path) -> str:
    """Get a readable name for the experiment from its directory path."""
    # Try to read hparams.txt for better naming
    hparams_file = exp_dir / "hparams.txt"
    if hparams_file.exists():
        try:
            with hparams_file.open("r", encoding="utf-8") as f:
                lines = f.readlines()
                # Create a name from key hyperparameters
                params = {}
                for line in lines:
                    if ":" in line:
                        key, value = line.strip().split(":", 1)
                        params[key.strip()] = value.strip()
                
                # Build a descriptive name
                name_parts = []
                if "model_name" in params:
                    name_parts.append(f"Model: {params['model_name']}")
                if "lr" in params:
                    name_parts.append(f"LR: {params['lr']}")
                if "local_epochs" in params:
                    name_parts.append(f"Epochs: {params['local_epochs']}")
                if "rounds" in params:
                    name_parts.append(f"Rounds: {params['rounds']}")
                
                if name_parts:
                    return " | ".join(name_parts)
        except Exception:
            pass
    
    # Fall back to directory name
    return exp_dir.name


def visualize_last_two_experiments(
    exp_root: str = "experiments",
    output_path: Optional[str] = None,
    show: bool = False
) -> Optional[Path]:
    """
    Create a single-page visualization comparing the last two experiments.
    
    Args:
        exp_root: Root directory containing experiments
        output_path: Path to save the visualization (if None, saves to exp_root/comparison.png)
        show: Whether to display the plot
    
    Returns:
        Path to saved visualization file, or None if not enough experiments found
    """
    # Find experiment directories
    experiments = find_experiment_directories(exp_root)
    
    if len(experiments) == 0:
        print("No experiments found!")
        return None
    
    if len(experiments) == 1:
        print(f"Only one experiment found: {experiments[0][0]}")
        print("Need at least two experiments for comparison.")
        return None
    
    # Get the last two experiments
    exp1_dir, exp1_time = experiments[0]
    exp2_dir, exp2_time = experiments[1]
    
    print(f"Experiment 1: {exp1_dir} (modified: {exp1_time})")
    print(f"Experiment 2: {exp2_dir} (modified: {exp2_time})")
    
    # Load metrics
    metrics1 = load_metrics(exp1_dir)
    metrics2 = load_metrics(exp2_dir)
    
    if metrics1 is None:
        print(f"Could not load metrics from {exp1_dir}")
        return None
    
    if metrics2 is None:
        print(f"Could not load metrics from {exp2_dir}")
        return None
    
    # Get experiment names
    exp1_name = get_experiment_name(exp1_dir)
    exp2_name = get_experiment_name(exp2_dir)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Federated Learning Experiments Comparison", fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy comparison
    ax1 = axes[0, 0]
    ax1.plot(metrics1["round"], metrics1["acc"], marker='o', label=exp1_name, linewidth=2, markersize=4)
    ax1.plot(metrics2["round"], metrics2["acc"], marker='s', label=exp2_name, linewidth=2, markersize=4)
    ax1.set_title("Test Accuracy Comparison", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Round", fontsize=10)
    ax1.set_ylabel("Accuracy", fontsize=10)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # Plot 2: Loss comparison
    ax2 = axes[0, 1]
    ax2.plot(metrics1["round"], metrics1["loss"], marker='o', label=exp1_name, linewidth=2, markersize=4, color='#1f77b4')
    ax2.plot(metrics2["round"], metrics2["loss"], marker='s', label=exp2_name, linewidth=2, markersize=4, color='#ff7f0e')
    ax2.set_title("Test Loss Comparison", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Round", fontsize=10)
    ax2.set_ylabel("Loss", fontsize=10)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Accuracy - individual experiments
    ax3 = axes[1, 0]
    ax3.plot(metrics1["round"], metrics1["acc"], marker='o', linewidth=2, markersize=4, label=exp1_name)
    ax3.set_title(f"Experiment 1: Accuracy", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Round", fontsize=10)
    ax3.set_ylabel("Accuracy", fontsize=10)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)
    
    # Add final accuracy annotation
    if metrics1["acc"]:
        final_acc1 = metrics1["acc"][-1]
        ax3.annotate(f'Final: {final_acc1:.2%}', 
                    xy=(metrics1["round"][-1], final_acc1),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                    fontsize=9)
    
    # Plot 4: Accuracy - second experiment
    ax4 = axes[1, 1]
    ax4.plot(metrics2["round"], metrics2["acc"], marker='s', linewidth=2, markersize=4, color='#ff7f0e', label=exp2_name)
    ax4.set_title(f"Experiment 2: Accuracy", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Round", fontsize=10)
    ax4.set_ylabel("Accuracy", fontsize=10)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)
    
    # Add final accuracy annotation
    if metrics2["acc"]:
        final_acc2 = metrics2["acc"][-1]
        ax4.annotate(f'Final: {final_acc2:.2%}', 
                    xy=(metrics2["round"][-1], final_acc2),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5),
                    fontsize=9)
    
    # Add summary text box
    summary_text = f"""
    Summary:
    Exp 1: {exp1_name}
      Final Acc: {metrics1['acc'][-1]:.2%} | Final Loss: {metrics1['loss'][-1]:.4f}
      Rounds: {len(metrics1['round'])}
    
    Exp 2: {exp2_name}
      Final Acc: {metrics2['acc'][-1]:.2%} | Final Loss: {metrics2['loss'][-1]:.4f}
      Rounds: {len(metrics2['round'])}
    """
    
    fig.text(0.5, 0.02, summary_text.strip(), ha='center', va='bottom', 
             fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.08, 1, 0.98])
    
    # Save the figure
    if output_path is None:
        output_path = Path(exp_root) / "comparison_last_two.png"
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return output_path


if __name__ == "__main__":
    import sys
    
    exp_root = sys.argv[1] if len(sys.argv) > 1 else "experiments"
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = visualize_last_two_experiments(exp_root=exp_root, output_path=output_path, show=False)
    
    if result:
        print(f"✓ Successfully created comparison visualization")
        sys.exit(0)
    else:
        print(f"✗ Failed to create visualization")
        sys.exit(1)

