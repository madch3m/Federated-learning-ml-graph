from __future__ import annotations
import sys, csv, itertools
from dataclasses import replace, asdict
from pathlib import Path
import importlib
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

m = importlib.import_module("main")
from model.cnn import CNN as BaselineCNN
from utils.experiments import create_experiment_dir
from utils.plotter import plot_history

EXP_ROOT = Path(__file__).resolve().parent
SUMMARY_CSV = EXP_ROOT / "summary_depth_clean.csv"

# Baseline is (s1=1, s2=1); we vary ONLY extra convs before each existing MaxPool.
GRID = {
    "stage1_convs": [1, 2, 3],   # baseline=1
    "stage2_convs": [1, 2, 3],   # baseline=1
}

def combos(grid: dict):
    keys = list(grid.keys())
    vals = [grid[k] for k in keys]
    cs = [{k: v for k, v in zip(keys, tup)} for tup in itertools.product(*vals)]
    return keys, cs

def make_depth_variant(stage1_convs: int, stage2_convs: int) -> nn.Module:
    c1, c2 = 32, 64  # same channels as baseline
    layers = []
    in_ch = 1
    # Stage 1: conv x n, then MaxPool(2)
    layers += [nn.Conv2d(in_ch, c1, 3, padding=1), nn.ReLU(inplace=True)]
    for _ in range(stage1_convs - 1):
        layers += [nn.Conv2d(c1, c1, 3, padding=1), nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(2)]
    # Stage 2: conv x n, then MaxPool(2)
    layers += [nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU(inplace=True)]
    for _ in range(stage2_convs - 1):
        layers += [nn.Conv2d(c2, c2, 3, padding=1), nn.ReLU(inplace=True)]
    layers += [nn.MaxPool2d(2)]
    # Head identical to baseline
    layers += [nn.Flatten(), nn.Linear(c2 * 7 * 7, 128), nn.ReLU(inplace=True), nn.Linear(128, 10)]
    return nn.Sequential(*layers)

def time_to_threshold(rounds, acc, thr=0.95):
    for r, a in zip(rounds, acc):
        if a >= thr:
            return r
    return float("inf")

def run_one(cfg_hp: m.HParams, dcfg: dict, run_name: str) -> dict:
    if dcfg["stage1_convs"] == 1 and dcfg["stage2_convs"] == 1:
        m.CNN = BaselineCNN
    else:
        m.CNN = lambda: make_depth_variant(dcfg["stage1_convs"], dcfg["stage2_convs"])
    m.hp = cfg_hp
    history, model = m.orchestrate()
    meta = {**asdict(cfg_hp), **dcfg, "variant": "baseline" if dcfg=="baseline" else "depth_clean"}
    exp_dir = create_experiment_dir(meta, exp_root=str(EXP_ROOT), name=run_name)
    torch.save(model.state_dict(), exp_dir / "mnist_cnn.pt")
    plot_history(history, exp_dir=exp_dir, show=False)

    R, A, L = history["round"], history["acc"], history["loss"]
    return {
        "exp_dir": str(exp_dir.relative_to(ROOT)),
        "rounds": R[-1] if R else 0,
        "final_acc": A[-1] if A else 0.0,
        "final_loss": L[-1] if L else 0.0,
        "best_acc": max(A) if A else 0.0,
        "best_acc_round": (A.index(max(A)) + 1) if A else 0,
        "t95": time_to_threshold(R, A, 0.95),
        **dcfg,
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
    keys, cs = combos(GRID)
    rows = []
    header = ["exp_dir","rounds","final_acc","final_loss","best_acc","best_acc_round","t95"] + keys
    for i, params in enumerate(cs, 1):
        cfg_hp = replace(m.HParams())
        run_name = f'run_{i:03d}_s1={params["stage1_convs"]}_s2={params["stage2_convs"]}'
        run_dir = EXP_ROOT / run_name
        if (run_dir / "metrics.csv").exists():
            result = {"exp_dir": str(run_dir.relative_to(ROOT)), **params}
        else:
            result = run_one(cfg_hp, params, run_name)
        rows.append(result)
        write_summary(rows, header)
    print(SUMMARY_CSV.relative_to(ROOT))

if __name__ == "__main__":
    main()