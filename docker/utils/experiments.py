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