from __future__ import annotations
import sys, csv, itertools
from dataclasses import replace, asdict
from pathlib import Path
import importlib
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

m = importlib.import_module("main")
from utils.experiments import create_experiment_dir
from utils.plotter import plot_history

EXP_ROOT = Path(__file__).resolve().parent

PARAM_GRID = {
    "lr": [0.1, 0.01],
    "local_epochs": [1, 2],
    "sample_clients": [0.25, 0.5],
    "dirichlet_alpha": [0.1, 0.5],
}

SUMMARY_CSV = EXP_ROOT / "grid_summary.csv"

def param_combos(grid: dict) -> tuple[list[str], list[dict]]:
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    combos = [{k: v for k, v in zip(keys, tup)} for tup in itertools.product(*vals)]
    return keys, combos

def run_one(cfg: m.HParams, run_name: str) -> dict:
    m.hp = cfg
    history, model = m.orchestrate()
    exp_dir = create_experiment_dir(asdict(cfg), exp_root=str(EXP_ROOT), name=run_name)
    torch.save(model.state_dict(), exp_dir / "mnist_cnn.pt")
    plot_history(history, exp_dir=exp_dir, show=False)
    rounds, accs, losses = history["round"], history["acc"], history["loss"]
    return {
        "exp_dir": str(exp_dir.relative_to(ROOT)),
        "rounds": rounds[-1] if rounds else 0,
        "final_acc": accs[-1] if accs else 0.0,
        "final_loss": losses[-1] if losses else 0.0,
        "best_acc": max(accs) if accs else 0.0,
        "best_acc_round": (accs.index(max(accs)) + 1) if accs else 0,
    }

def write_summary(rows: list[dict], header: list[str]) -> None:
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            for k in header:
                r.setdefault(k, "")
            w.writerow(r)

def main():
    keys, combos = param_combos(PARAM_GRID)
    header = ["exp_dir", "rounds", "final_acc", "final_loss", "best_acc", "best_acc_round"] + keys
    rows: list[dict] = []
    for i, params in enumerate(combos, 1):
        cfg = replace(m.HParams(), **params)
        run_name = f"run_{i:03d}_" + "_".join(f"{k}={params[k]}" for k in keys)
        run_dir = EXP_ROOT / run_name
        if (run_dir / "metrics.csv").exists():
            result = {"exp_dir": str(run_dir.relative_to(ROOT))}
        else:
            result = run_one(cfg, run_name)
        rows.append({**result, **params})
        write_summary(rows, header)
    print(SUMMARY_CSV.relative_to(ROOT))

if __name__ == "__main__":
    main()