import csv, argparse
from pathlib import Path
import matplotlib.pyplot as plt

def read_summary(path: Path):
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        r=csv.DictReader(f)
        for row in r:
            rows.append({
                "exp_dir": row.get("exp_dir",""),
                "final_acc": float(row.get("final_acc",0) or 0),
                "final_loss": float(row.get("final_loss",0) or 0),
                "best_acc": float(row.get("best_acc",0) or 0),
            })
    return rows

def read_metrics(run_dir: Path):
    rounds, acc, loss = [], [], []
    p = run_dir / "metrics.csv"
    if not p.exists(): return rounds, acc, loss
    with p.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rounds.append(int(row["round"]))
            acc.append(float(row["test_accuracy"]))
            loss.append(float(row["test_loss"]))
    return rounds, acc, loss

def run_id(name: str) -> str:
    if name.startswith("run_"):
        parts = name.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            return "_".join(parts[:2])
    return name[:10]

def time_to_threshold(rounds, acc, thr=0.95):
    for r, a in zip(rounds, acc):
        if a >= thr:
            return r
    return float("inf")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="experiments/grid_lr_epochs")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.95)
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    rows = read_summary(exp_dir / "grid_summary.csv")

    runs = []
    for row in rows:
        full = Path(row["exp_dir"]).name or ""
        rid = run_id(full)
        run_dir = exp_dir / full if (exp_dir / full).exists() else Path(row["exp_dir"]).resolve()
        R, A, L = read_metrics(run_dir)
        if not R: continue
        tthr = time_to_threshold(R, A, args.thr)
        runs.append({
            "rid": rid, "full": full, "dir": run_dir,
            "rounds": R, "acc": A, "loss": L,
            "tthr": tthr, "best": max(A),
            "final_loss": row["final_loss"],
        })

    runs.sort(key=lambda c: (c["tthr"], -c["best"]))

    # color map for ALL runs
    palette = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = {r["rid"]: palette[i % len(palette)] for i, r in enumerate(runs)}

    # ---- Plot 1: accuracy truncated at threshold (top N), full y-axis [0,1] ----
    curves = runs[:args.top]
    plt.figure(figsize=(10,6))
    for c in curves:
        R, A = c["rounds"], c["acc"]
        if c["tthr"] != float("inf"):
            idx = next(i for i,(r,a) in enumerate(zip(R,A)) if a >= args.thr)
            R, A = R[:idx+1], A[:idx+1]
        lbl = f'{c["rid"]} (t{int(args.thr*100)}={ "∞" if c["tthr"]==float("inf") else c["tthr"]})'
        plt.plot(R, A, label=lbl, linewidth=2, alpha=0.9, color=colors[c["rid"]])
    plt.xlabel("Round"); plt.ylabel("Test Accuracy")
    plt.title(f"Accuracy vs Round (truncated at {int(args.thr*100)}%)")
    plt.ylim(0.0, 1.0)  # full axis
    plt.legend(fontsize=8, ncol=1, loc="upper left", bbox_to_anchor=(1.02,1))
    plt.tight_layout()
    plt.savefig(exp_dir / "grid_acc_curves_t95.png", dpi=160, bbox_inches="tight")
    plt.close()

    # ---- Plot 2: final loss for ALL runs (matching colors) ----
    labels = [r["rid"] for r in runs]
    losses = [r["final_loss"] for r in runs]
    plt.figure(figsize=(max(10, 0.8*len(labels)), 5))
    bars = plt.bar(range(len(labels)), losses)
    for i, b in enumerate(bars):
        b.set_color(colors[labels[i]])
    step = max(1, len(labels)//25)
    plt.xticks(range(0, len(labels), step), [labels[i] for i in range(0,len(labels),step)],
               rotation=45, ha="right", fontsize=8)
    plt.ylabel("Final Test Loss")
    plt.title("Final Test Loss (all runs)")
    plt.tight_layout()
    plt.savefig(exp_dir / "grid_final_loss_all.png", dpi=160, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()