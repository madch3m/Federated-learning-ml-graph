from __future__ import annotations
from pathlib import Path
from datetime import datetime

def create_experiment_dir(hparams: dict, exp_root: str = "experiments", name: str | None = None) -> Path:
    """
    Creates experiments/<name>/, writes hparams.txt, and returns the path.
    If name is None, auto-generates a descriptive timestamped name.
    """
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    if name is None:
        parts = [ts]
        for key in ("num_clients", "iid", "dirichlet_alpha", "local_epochs", "rounds", "lr", "batch"):
            if key in hparams:
                parts.append(f"{key}={hparams[key]}")
        name = "_".join(map(str, parts))

    exp_dir = Path(exp_root) / name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # hparams.txt (sorted, key: value)
    with (exp_dir / "hparams.txt").open("w", encoding="utf-8") as f:
        for k in sorted(hparams.keys()):
            f.write(f"{k}: {hparams[k]}\n")

    return exp_dir


def visualize_last_two_experiments(exp_root: str = "experiments", output_path: str | None = None) -> Path | None:
    """
    Convenience function to visualize the last two experiments.
    Imports and calls the visualization function from visualize_experiments module.
    
    Args:
        exp_root: Root directory containing experiments
        output_path: Path to save the visualization (if None, saves to exp_root/comparison_last_two.png)
    
    Returns:
        Path to saved visualization file, or None if not enough experiments found
    """
    try:
        from .visualize_experiments import visualize_last_two_experiments as _visualize
        return _visualize(exp_root=exp_root, output_path=output_path, show=False)
    except ImportError:
        # Fallback if import fails
        import sys
        from pathlib import Path
        utils_dir = Path(__file__).parent
        sys.path.insert(0, str(utils_dir.parent))
        from utils.visualize_experiments import visualize_last_two_experiments as _visualize
        return _visualize(exp_root=exp_root, output_path=output_path, show=False)